import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub
import dagshub
import mlflow.client
dagshub.init(repo_owner='manikantmnnit', repo_name='swiggy-delivery-time-prediction', mlflow=True)


# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/swiggy-delivery-time-prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info

def assign_alias_to_stage(model_name, stage, alias):
    """
    Assign an alias to the latest version of a registered model within a specified stage and update its stage to "Staging".

    :param model_name: The name of the registered model.
    :param stage: The stage of the model version for which the alias is to be assigned. Can be
                  "Production", "Staging", "Archived", or "None".
    :param alias: The alias to assign to the model version.
    :return: None
    """
    # Get the latest model version for the specified stage
    latest_mv = client.get_latest_versions(model_name, stages=[stage])[0]
    
    # Update the stage of the model to "Staging"
    client.transition_model_version_stage(
        name=model_name,
        version=latest_mv.version,
        stage="Staging"
    )
    
    # Assign an alias to the model version
    client.set_registered_model_alias(model_name, alias, latest_mv.version)


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    run_info = load_model_information(run_info_path)
    
    # get the run id
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    
    # model to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"
    
    
    # register the model
    model_version = mlflow.register_model(model_uri=model_registry_path,
                                          name=model_name)  # register the model
    
    
    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")
    
    # update the stage of the model to staging
    client = MlflowClient() # create a mlflow client
    
   # Example usage
    assign_alias_to_stage(model_name=registered_model_name, stage="Staging", alias="staging")
  
    
    logger.info("Model pushed to Staging stage")
    