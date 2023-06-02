import pytest
import tempfile
from pathlib import Path
from interpolation.interpolation import exponential_decay_interpolation

# Define your parametrized fixture
@pytest.fixture(params=[
    ("start_image1.png", "end_image1.png", 5),
    ("start_image2.png", "end_image2.png", 10),
    ("start_image3.png", "end_image3.png", 7),
])
def image_data(request):
    print("setup")
    temp_dir = tempfile.TemporaryDirectory()
    print(temp_dir.name)
    yield { 'test_data': request.param, 'dest_dir': temp_dir }
    print("teardown")
    temp_dir.cleanup()

# Modify your test functions to request the new fixture
def test_interpolation_image_count(image_data):
    print(f"test_interpolation_image_count: image_data: {image_data}")
    start_image_path, end_image_path, num_interpolated_frames = image_data['test_data']
    dest_dir = image_data['dest_dir']
    exponential_decay_interpolation(dest_dir, start_image_path, end_image_path, num_interpolated_frames)
    assert True

def test_interpolation_image_metadata(image_data):
    start_image_path, end_image_path, num_interpolated_frames = image_data['test_data']
    dest_dir = image_data['dest_dir']
    exponential_decay_interpolation(dest_dir, start_image_path, end_image_path, num_interpolated_frames)
    assert True
