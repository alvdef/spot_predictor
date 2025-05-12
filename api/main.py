from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
import boto3
import time
import pandas as pd
import numpy as np
import torch
import yaml
import warnings

from utils import extract_time_features, get_device
from model import FeatureSeq2Seq, FeatureSeq2Seq_v2


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    """
    with path.open() as f:
        return yaml.safe_load(f)


class PredictRequest(BaseModel):
    region: str = Field(..., description="AWS region, e.g., eu-north-1")
    av_zone: str = Field(..., description="Availability zone suffix, e.g., 'a', 'b'")
    instance_type: str = Field(..., description="EC2 instance type, e.g., m6i.large")
    os: str = Field(..., description="Product description, e.g., 'Linux/UNIX'")
    days: int = Field(..., gt=0, description="Number of days of history to retrieve")


class PredictResponse(BaseModel):
    predictions: List[float]
    input_parameters: Dict[str, Any]
    processing_time: float = Field(..., description="Time taken in seconds")


app = FastAPI(title="Spot Price Predictor API")


# On startup: load config, model, and instance features
@app.on_event("startup")
def startup_event():
    global config, model, features_tensor, feature_mapping, instance_info_df, seq_cfg, device
    api_dir = Path(__file__).parent

    # Load API config
    config = load_yaml(api_dir / "config.yaml")
    seq_cfg = config.get("sequence_config", {})

    # Load instance info DataFrame
    instance_info_df = pd.read_pickle(api_dir / "instance_info_df.pkl")

    # Precompute instance feature tensors based on config
    feature_cols = config.get("dataset_config", {}).get("instance_features", [])
    filtered = instance_info_df[feature_cols].copy()
    # build DataFrame of one-hot and binary features
    features_df = pd.DataFrame(index=filtered.index)
    for col in feature_cols:
        vals = filtered[col]
        # detect list-type
        if vals.apply(lambda x: isinstance(x, list)).any():
            uniques = set(
                val for lst in vals.dropna() if isinstance(lst, list) for val in lst
            )
            for u in uniques:
                features_df[f"{col}_{u}"] = vals.apply(
                    lambda x: True if isinstance(x, list) and u in x else False
                )
        else:
            dummies = pd.get_dummies(vals, prefix=col)
            features_df = pd.concat([features_df, dummies], axis=1)
    # create tensor and mapping
    device = get_device()
    features_tensor = torch.tensor(
        features_df.values, dtype=torch.float32, device=device
    )
    feature_mapping = {iid: idx for idx, iid in enumerate(features_df.index)}

    # Load model
    model_type = config.get("model_config", {}).get("model_type")
    # pick class
    model_cls = {
        "FeatureSeq2Seq": FeatureSeq2Seq,
        "FeatureSeq2Seq_v2": FeatureSeq2Seq_v2,
    }.get(model_type)
    if model_cls is None:
        raise RuntimeError(f"Unknown model type {model_type}")
    model = model_cls(str(api_dir))
    model.load()
    model.to(device)
    model.eval()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Fetch recent spot price history, process features, and forecast future prices.
    """
    start_time = time.time()

    # Build AWS client
    client = boto3.client("ec2", region_name=req.region)

    # Determine time range
    end_time = pd.Timestamp.utcnow()
    start_time_hist = end_time - pd.Timedelta(days=req.days)

    # Retry logic
    max_retries = config.get("api", {}).get("max_retries", 3)
    resp = {}
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.describe_spot_price_history(
                StartTime=start_time_hist.to_pydatetime(),
                EndTime=end_time.to_pydatetime(),
                InstanceTypes=[req.instance_type],
                ProductDescriptions=[req.os],
                AvailabilityZone=f"{req.region}{req.av_zone}",
            )
            break
        except Exception as e:
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail=str(e))
            time.sleep(2**attempt)

    # Convert to DataFrame
    history = resp.get("SpotPriceHistory", [])
    if not history:
        raise HTTPException(status_code=404, detail="No spot price history returned")
    df = pd.DataFrame(history)
    df = df.rename(columns={"Timestamp": "price_timestamp", "SpotPrice": "spot_price"})
    df["price_timestamp"] = pd.to_datetime(df["price_timestamp"], utc=True)

    # Identify instance ID
    mask = (
        (instance_info_df["region"] == req.region)
        & (instance_info_df["av_zone"] == req.av_zone)
        & (instance_info_df["instance_type"] == req.instance_type)
        & (instance_info_df["product_description"] == req.os)
    )
    matches = instance_info_df[mask]
    if len(matches) != 1:
        raise HTTPException(
            status_code=400, detail="Instance metadata match not found or ambiguous"
        )
    instance_id = matches.index[0]
    df["id_instance"] = instance_id

    # Group to fixed hours and forward-fill missing
    th = seq_cfg.get("timestep_hours", 1)
    df["price_timestamp"] = df["price_timestamp"].dt.floor(f"{th}h")
    # build full time index
    full_idx = pd.date_range(
        start=df["price_timestamp"].min(),
        end=df["price_timestamp"].max(),
        freq=f"{th}h",
        tz="UTC",
    )
    full_df = pd.DataFrame({"price_timestamp": full_idx})
    full_df["id_instance"] = instance_id
    merged = pd.merge(
        full_df,
        df[["price_timestamp", "id_instance", "spot_price"]],
        on=["price_timestamp", "id_instance"],
        how="left",
    )
    merged["spot_price"] = merged["spot_price"].ffill()

    # Build sequence of last N timesteps
    seq_len = int(req.days * 24 / th)
    seq_values = merged["spot_price"].values[-seq_len:].astype(np.float32)

    # Time features
    tf_list = extract_time_features(
        merged["price_timestamp"], config["dataset_config"]["time_features"]
    )
    tf_array = np.array(tf_list, dtype=np.float32)[-seq_len:]

    # Convert to tensors
    seq_tensor = (
        torch.tensor(seq_values, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    tf_tensor = torch.tensor(tf_array, dtype=torch.float32, device=device).unsqueeze(0)
    feat_idx = feature_mapping[instance_id]
    inst_tensor = features_tensor[feat_idx].unsqueeze(0)

    # Forecast
    n_steps = seq_len
    preds = model.forecast((seq_tensor, inst_tensor, tf_tensor), n_steps, [instance_id])
    preds = preds.cpu().numpy().flatten().tolist()

    processing_time = time.time() - start_time
    return PredictResponse(
        predictions=preds,
        input_parameters=req.dict(),
        processing_time=processing_time,
    )
