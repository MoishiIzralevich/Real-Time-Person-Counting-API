# Real-Time Person Counting API

## Overview

This FastAPI application processes video files to detect and count unique persons in real-time. It uses the YOLOv8 model to detect persons and optionally counts how many males and females are detected in the video. The real-time count is streamed back to the client during processing.

### Key Features:
- Detect and count unique persons in the video.
- Real-time tracking of unique persons.
- Optionally count how many males and females are detected.
- Optionally save output file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MoishiIzralevich/Real-Time-Person-Counting-API.git
   cd Real-Time-Person-Counting-API
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

To start the FastAPI server locally, run:

```bash
uvicorn app:app --reload
```

This will start the server at `http://127.0.0.1:8000`.

### Endpoint: `/process-video`

- **Method**: `POST`
- **Request**: Video file in the form-data.
- **Query Parameter**: 

  `use_gender_classification` (optional, default `false`): If set to `true`, gender classification (Male/Female) will be applied.

  `save_output` (optional, default `false`): If set to `true`, the processed video will be saved and the output video path will be included in the response.
  
#### Example Request

- **URL**: `http://127.0.0.1:8000/process-video?use_gender_classification=true`
- **Body**: Upload a video file in the form-data.

#### Example Request (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/process-video/?use_gender_classification=false&save_output=true" 
    -H "accept: application/json" 
    -H "Content-Type: multipart/form-data" 
    -F "file=@./videos/police_1_minute.mp4"
```

#### Example Response:

```json
{
  "message": "Video processed successfully. Gender classification applied.",
  "output_video_path": "output_processed_video.mp4",
  "total_unique_persons": 5,
  "male_count": 2,
  "female_count": 3
}
```

The response will contain:
- A message indicating success.
- The path to the processed video.
- The total count of unique persons.
- The count of males and females (if gender classification was applied).

## Code Structure

- **app.py**: FastAPI application file with the `/process-video` endpoint.
- **models/**: Contains the YOLO and gender classification models.
- **videos/**: Contains the videos for testing.
- **tests/**: Contains the testing file.
- **requirements.txt**: Lists the project dependencies.

## Tests

To run the tests, use the following command:

```bash
pytest
```

The tests ensure that the API processes videos correctly and returns accurate results based on the given parameters.