# inspect_h5.py
# Opens a single downloaded NISAR HDF5 file and prints its full internal
# directory structure — group names, dataset shapes, dtypes, and attributes.
# Use this to verify HDF5 paths before referencing them in download_nisar.py.

import h5py  # HDF5 file reader


# =============================================================================
# Parameters
# Set H5_FILE to the path of any downloaded NISAR HDF5 file.
# =============================================================================
H5_FILE = "./data/NISAR_L2_PR_GUNW_004_073_D_054_005_4000_SH_20251103T062351_20251103T062427_20251115T062351_20251115T062427_X05010_N_F_J_001.h5"  # path to the HDF5 file to inspect (edit this)


# =============================================================================
# Visitor function
# h5py.File.visititems() calls this function once for every item (group or
# dataset) in the file, passing its full internal path and the h5py object.
# Defined as a function because visititems() requires a callable.
# =============================================================================
def print_item(name, obj):
    """Print name, type, shape/dtype (datasets), and attributes for one HDF5 item."""
    indent = "  " * name.count("/")               # indent depth matches the hierarchy level
    if isinstance(obj, h5py.Dataset):              # item is a dataset (holds array data)
        print(f"{indent}[DATASET]  {name}")        # print full internal path
        print(f"{indent}           shape : {obj.shape}")   # array dimensions
        print(f"{indent}           dtype : {obj.dtype}")   # element data type
    elif isinstance(obj, h5py.Group):              # item is a group (acts like a directory)
        print(f"{indent}[GROUP]    {name}")        # print full internal path

    if obj.attrs:                                  # only print attributes section if any exist
        for attr_name, attr_val in obj.attrs.items():   # iterate over all attributes on this item
            print(f"{indent}           @{attr_name}: {attr_val}")  # print attribute name and value


# =============================================================================
# Open file and walk structure
# visititems() traverses every group and dataset in depth-first order and
# calls print_item() for each one.  The root group ("/") is not visited by
# visititems, so it is printed separately first.
# =============================================================================
print(f"Inspecting: {H5_FILE}\n")          # log the file being inspected
with h5py.File(H5_FILE, "r") as hf:        # open file in read-only mode
    print(f"[ROOT]     /")                  # print the root group manually
    if hf.attrs:                            # print root-level attributes if any exist
        for attr_name, attr_val in hf.attrs.items():   # iterate root attributes
            print(f"           @{attr_name}: {attr_val}")  # print each root attribute
    hf.visititems(print_item)               # walk and print every item below root

print("\nDone.")  # signal that the full tree has been printed
