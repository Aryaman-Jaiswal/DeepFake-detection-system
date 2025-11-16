# A Robust Deepfake Detection System Using a Classifier Ensemble

This project is a full-stack web application designed to detect deepfake videos. It uses a backend built with Python and Flask to serve a PyTorch model ensemble, and a frontend built with React to provide a user-friendly interface for uploading and analyzing videos.

## Project Structure

The repository is organized into two main parts: the `backend` server and the `frontend` application.

```
deepfake-detector-website/
|
|-- backend/
|   |-- app.py                     # Main Flask server code
|   |-- blazeface.py               # BlazeFace model definition code
|   |-- anchors.npy                # BlazeFace anchor file
|   |-- blazeface.pth              # BlazeFace pre-trained weights file
|   |-- requirements.txt           # Backend Python dependencies
|   |-- models/
|       |-- xception_weights.pth       # Your trained XceptionNet weights
|       |-- efficientnet_b4_weights.pth# Your trained EfficientNet weights
|
|   |-- uploads/                   # (This folder will be created automatically by the app)
|
|-- frontend/
|   |-- package.json               # Frontend dependencies and scripts
|   |-- src/
|       |-- App.js                 # Main React component for the UI
|       |-- App.css                # Styles for the component
|-- node_modules/                  # (This folder will be created by npm install)
```

## How to Run the Application

To run this project, you will need to have two separate terminals open: one for the backend server and one for the frontend application.

### 1. Backend Setup (Flask Server)

**Prerequisites:** Python 3.x and `pip` must be installed.

```bash
# 1. Open a new terminal and navigate to the backend directory
cd backend

# 2. Install the required Python packages from the requirements file
pip install -r requirements.txt

# 3. Run the Flask server
python app.py
```

The backend server will start up, load the AI models, and begin listening for requests on http://127.0.0.1:5000. You should see a "Setup complete. Server is ready." message. Leave this terminal running.

### 2. Frontend Setup (React App)

**Prerequisites:** Node.js and npm must be installed.

```bash
# 1. Open a second, new terminal and navigate to the frontend directory
cd frontend

# 2. Install all project dependencies from the package.json file
# This may take a few minutes
npm install

# 3. Start the React development server
npm start
```

This command will automatically open the web application in your default browser at http://localhost:3000.  
You can now use the interface to upload an MP4 video and see the prediction from the model ensemble.

Note: The trained models (.pth files) in backend/models/ and the blazeface.pth file are included in this repository to allow the application to run directly without needing to be re-trained.
