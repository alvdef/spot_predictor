


region_prices_df = read_prices_files('eu-central-1')

id_instances = region_prices_df["id_instance"].unique()
complete_time_df = generate_complete_time_df(id_instances)

region_prices_df = merge_prices_with_time(region_prices_df, complete_time_df)


# For Torch
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        context_len=2048,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)


# Spliting into 94% and 6%
split_idx = int(len(df) * 0.94)
# Split the dataframe into train and test sets
train_df = df[:split_idx]
test_df = df[split_idx:]
print(train_df.shape, test_df.shape)


# Initialize the TimesFM model with specified parameters
tfm = timesfm.TimesFm(
   context_len=128,       # Length of the context window for the model
   horizon_len=24,        # Forecasting horizon length
   input_patch_len=32,    # Length of input patches
   output_patch_len=128,  # Length of output patches
   num_layers=20,        
   model_dims=1280,      
)
# Load the pretrained model checkpoint
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
# Forecasting the values using the TimesFM model
timesfm_forecast = tfm.forecast_on_df(
   inputs=train_df,       # Input training data for training
   freq="MS",             # Frequency of the time-series data
   value_name="y",        # Name of the column containing the values to be forecasted
   num_jobs=-1,           # Set to -1 to use all available cores
)
timesfm_forecast = timesfm_forecast[["ds","timesfm"]]
