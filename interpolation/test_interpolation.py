import pytest
import tempfile
from pathlib import Path
from interpolation.interpolation import exponential_decay_interpolation, basic_interpolation
import time

# Define your parametrized fixture
@pytest.fixture(params=[
    ("interpolation/data/start_image_1.png", "interpolation/data/end_image_1.png", 1),
    ("interpolation/data/start_image_1.png", "interpolation/data/end_image_1.png", 5),
    ("interpolation/data/start_image_1.png", "interpolation/data/end_image_1.png", 7),
    ("interpolation/data/start_image_1.png", "interpolation/data/end_image_1.png", 13),
])
def image_data(request):
    print("setup")
    start_image_str, _, _ = request.param

    start_image_path = Path(start_image_str)
    # create a directory in the same folder as start_image_path prefixed with a millisecond timestamp of format <start_image_path.stem>-<millis>
    temp_dir = start_image_path.parent / f"{start_image_path.stem}-{int(time.time_ns()*1000)}"

    # create the directory
    temp_dir.mkdir()

    yield { 'test_data': request.param, 'dest_dir': temp_dir }
    print("teardown")

    # while we want to view the temp dir we don't delete it. 
    # shutil.rmtree(temp_dir)

# Modify your test functions to request the new fixture
def test_interpolation_image_count(image_data):
    print(f"test_interpolation_image_count: image_data: {image_data}")
    start_image_path, end_image_path, num_interpolated_frames = image_data['test_data']
    dest_dir = image_data['dest_dir']
    exponential_decay_interpolation(dest_dir, start_image_path, end_image_path, num_interpolated_frames)
    assert True

def test_basic_interpolation(image_data):
    print(f"test_basic_interpolation: image_data: {image_data}")
    start_image_path, end_image_path, num_interpolated_frames = image_data['test_data']

    # check that the start and end images exist
    assert Path(start_image_path).exists()
    assert Path(end_image_path).exists()

    dest_dir_path = Path(image_data['dest_dir'])
    assert dest_dir_path.exists()
    basic_interpolation(dest_dir_path, start_image_path, end_image_path)
    assert True