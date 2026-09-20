import mlflow
import os
import shutil

def export_model():
    # Based on filesystem search: mlruns/1/models/m-e5b8ec269f0a4bc891ec899769216127 is the registered model
    source_path = "mlruns/1/models/m-e5b8ec269f0a4bc891ec899769216127/artifacts"
    export_path = "models/champion"

    print(f"Attempting to export model from: {source_path} to {export_path}...")
    
    try:
        if not os.path.exists(source_path):
            print(f"Error: Source path {source_path} does not exist.")
            return

        # Create clean directory
        if os.path.exists(export_path):
            print(f"Removing existing {export_path} directory...")
            shutil.rmtree(export_path)
        
        os.makedirs(export_path, exist_ok=True)
        
        # Copy all artifacts from the identified champion model
        for item in os.listdir(source_path):
            s = os.path.join(source_path, item)
            d = os.path.join(export_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
                
        print(f"Successfully exported champion model to: {export_path}")
        
    except Exception as e:
        print(f"Error exporting model: {e}")

if __name__ == "__main__":
    export_model()
