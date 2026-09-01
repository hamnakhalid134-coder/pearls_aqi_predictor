import hopsworks


# --------------------------------
# 1. Connect to Hopsworks
# --------------------------------

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_hk26",
    port=443,
    engine="python"
)

fs = project.get_feature_store()


# --------------------------------
# 2. Get historical feature group
# --------------------------------

fg = fs.get_feature_group(
    name="aqi_historical_features",
    version=1
)


# --------------------------------
# 3. Columns NOT used as model inputs
# --------------------------------

targets = [
    "aqi_24h",
    "aqi_48h",
    "aqi_72h"
]


# --------------------------------
# 4. Create one Feature View
#    per forecast horizon
# --------------------------------

for target in targets:

    # Keep all columns except the OTHER targets.
    other_targets = [
        col for col in targets
        if col != target
    ]

    query = fg.select_except(other_targets)

    view_name = target + "_view"

    fv = fs.get_or_create_feature_view(
        name=view_name,
        version=1,
        query=query,
        labels=[target],
        description=(
            f"Feature view for direct {target} "
            "AQI forecasting for Lahore."
        )
    )

    print(
        "Created Feature View:",
        fv.name,
        "Version:",
        fv.version
    )


print("\nAll three Feature Views created successfully!")