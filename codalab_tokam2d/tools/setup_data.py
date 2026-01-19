"""Fetcher for RAMP data stored in OSF

To adapt it for another challenge, change the CHALLENGE_NAME and upload
public/private data as `tar.gz` archives in dedicated OSF folders named after
the challenge.
"""
import sys
import h5py
import shutil
import tarfile
import argparse
import numpy as np
from zlib import adler32
from pathlib import Path
from osfclient.api import OSF
from osfclient.exceptions import UnauthorizedException

CHALLENGE_NAME = 'tokam2d'
PUBLIC_PROJECT = "cmbs5"
PRIVATE_PROJECT = "8u6bd"
PRIVATE_CKSUM = 2762212072
PUBLIC_CKSUM = 4201094694


def get_folder(code, token=None):
    "Get connection to OSF and info relative to public/private data."
    # Get the connection to OSF and find the folder in the OSF project
    osf = OSF(token=token)

    try:
        project = osf.project(code)
        store = project.storage('osfstorage')
    except UnauthorizedException:
        raise ValueError("Invalid credentials for RAMP private storage.")
    return get_one_element(store.folders, CHALLENGE_NAME)


def get_one_element(container, name):
    "Get one element from OSF container with a comprehensible failure error."
    elements = [f for f in container if f.name == name]
    container_name = (
        container.name if hasattr(container, 'name') else CHALLENGE_NAME
    )
    assert len(elements) == 1, (
        f'There is no element named {name} in {container_name} from the RAMP '
        'OSF account.'
    )
    return elements[0]


def hash_folder(folder_path):
    """Return the Adler32 hash of an entire directory."""
    folder = Path(folder_path)

    # Recursively scan the folder and compute a checksum
    checksum = 1
    for f in sorted(folder.rglob('*')):
        if f.is_dir():
            continue
        checksum = adler32(f.read_bytes(), checksum)

    return checksum


def checksum_data(data_dir, cksum, raise_error=False):
    print("Checking the data...", end='', flush=True)
    local_checksum = hash_folder(data_dir)
    if raise_error and cksum != local_checksum:
        raise ValueError(
            f"The checksum does not match. Expecting {cksum} but "
            f"got {local_checksum}. The archive seems corrupted. Try to "
            f"remove {data_dir} and re-run this command."
        )
    if cksum == local_checksum:
        print(" done.")
    else:
        print(" failed.")

    return local_checksum == cksum


def download_from_osf(folder, filename, data_dir=None):
    # Download the archive in the data
    if data_dir is None:
        data_dir = Path.cwd()
    target_path = data_dir / filename
    osf_file = get_one_element(folder.files, filename)
    print(f"Downloading {filename}...\r", end='', flush=True)
    with open(target_path, 'wb') as f:
        osf_file.write_to(f)
    print("Downloading done.".ljust(40))
    return target_path


def setup_data(data_path, private=False, token=None):
    "Download and uncompress the data from OSF."
    data_path = Path(data_path)
    cksum = PRIVATE_CKSUM if private else PUBLIC_CKSUM

    if not data_path.exists() or cksum is None:
        data_path.mkdir(exist_ok=True)
    elif checksum_data(data_path, cksum, raise_error=False):
        print("Data already downloaded and verified.")
        return

    # Download the public data
    if not private:
        archive = Path() / "public_dev_data.tar.gz"
        if not archive.exists():
            public_folder = get_folder(PUBLIC_PROJECT)
            archive = download_from_osf(public_folder, "public_dev_data.tar.gz")
        print("Extracting the data...", end='', flush=True)
        with tarfile.open(archive) as tar:
            tar.extractall(data_path.parent, filter='data')
        # archive.unlink()
        print(" done.")

    else:
        raw_data = Path() / "raw_data.tar.gz"
        if not raw_data.exists():
            private_folder = get_folder(PRIVATE_PROJECT, token=token)
            raw_data = download_from_osf(private_folder, "raw_data.tar.gz")
        print("Extracting the data...", end='', flush=True)
        with tarfile.open(raw_data) as tar:
            tar.extractall(filter='data')
        # raw_data.unlink()
        print(" done.")
        raw_path = Path() / "raw_data"

        # Setup data structure
        print("Setting up the data...", end='', flush=True)
        data_path.mkdir(exist_ok=True)
        input_data = data_path / "input_data"
        for sub_folder in ["train", "test", "private_test"]:
            (input_data / sub_folder).mkdir(parents=True, exist_ok=True)
        ref_data = data_path / "reference_data"
        ref_data.mkdir(exist_ok=True)

        # Setup train data
        for f in ["blob_i", "blob_dwi"]:
            for ext in [".xml", ".h5"]:
                fname = f"{f}{ext}"
                shutil.move(
                    str(raw_path / fname), str(input_data / "train" / fname)
                )
        fname = "turb_i.h5"
        shutil.move(
            str(raw_path / fname), str(input_data / "train" / fname)
        )

        # Setup test data
        fname = raw_path / "turb_dwi.h5"
        from ingestion_program.tokam2d_utils.xml_loader import XMLLoader
        from ingestion_program.tokam2d_utils.xml_loader import dump_to_xml
        annotations = XMLLoader(fname.with_suffix('.xml'))()
        annotations = {int(k.split('-')[1]): v for k, v in annotations.items()}
        with h5py.File(fname, 'r') as f:
            indices = f['indices']
            n_test = len(annotations) // 2
            test_idx, private_idx = indices[:n_test], indices[n_test:]
            dump_to_xml([
                {
                    "frame_index": f"test-{idx}",
                    "boxes": box,
                    "scores": np.array([1.0] * len(box))
                }
                for idx, box in annotations.items() if idx in test_idx
            ], ref_data / "test_labels.xml")
            with h5py.File(input_data / "test" / "test.h5", "w") as g:
                g['density'] = f['density'][:n_test]
                g['indices'] = test_idx

            dump_to_xml([
                {
                    "frame_index": f"private_test-{idx}",
                    "boxes": box,
                    "scores": np.array([1.0] * len(box))
                }
                for idx, box in annotations.items() if idx in private_idx
            ], ref_data / "private_test_labels.xml")
            with h5py.File(input_data / "private_test" / "private_test.h5", "w") as g:
                g['density'] = f['density'][n_test:]
                g['indices'] = private_idx

        (raw_path / "turb_dwi.h5").unlink()
        (raw_path / "turb_dwi.xml").unlink()
        raw_path.rmdir()
        print(" done.")

    checksum_data(data_path, cksum, raise_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f'Data loader for the {CHALLENGE_NAME} challenge on RAMP.'
    )
    parser.add_argument('--data-path', type=Path, default=Path("dev_phase"),
                        help='If this flag is used, download the private data '
                        'from OSF. This requires the username and password '
                        'options to be provided.')
    parser.add_argument('--private', action="store_true",
                        help='If this flag is used, download the private data '
                        'from OSF. This requires the username and password '
                        'options to be provided.')
    parser.add_argument(
        '--token', type=str, default=None,
        help="Token to access OSF private repo. Can be generated from "
        "https://osf.io/settings/tokens"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

    setup_data(args.data_path, private=args.private, token=args.token)
