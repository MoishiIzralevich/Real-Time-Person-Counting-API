import pytest
import httpx
import os

# Define the URL of the FastAPI application
URL = "http://127.0.0.1:8000/process-video/"

@pytest.mark.asyncio
async def test_process_video():
    # Prepare the path to the video file
    video_file_path = "../videos/police_1_minute.mp4"  # Ensure the path is correct
    assert os.path.exists(video_file_path), "Video file does not exist."

    # Open the video file
    with open(video_file_path, "rb") as video_file:
        # Prepare the files and parameters for the POST request
        files = {
            "file": ("police_1_minute.mp4", video_file, "video/mp4")
        }

        # First test without gender classification
        params = {
            "use_gender_classification": "true",  # Use 'false' for gender classification
            "save_output": "true"  # Save the output video
        }

        # Send the POST request using httpx with a custom timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(450.0)) as client:
            response = await client.post(URL, files=files, params=params)

            # Check if the request was successful
            assert response.status_code == 200, f"Request failed with status code {response.status_code}"

            # Parse the response JSON
            response_json = response.json()

           
            # Verify the response matches the expected structure and values
            assert "message" in response_json
            assert response_json["message"] == "Video processed successfully. Gender classification applied."
            assert "output_video_path" in response_json
            assert response_json["output_video_path"] == "output_processed_video.mp4"
            assert "total_unique_persons" in response_json
            assert response_json["total_unique_persons"] == 31
            assert "male_count" in response_json
            assert response_json["male_count"] == 13
            assert "female_count" in response_json
            assert response_json["female_count"] == 8
