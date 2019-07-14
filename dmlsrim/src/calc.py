"""."""
from srim import Ion, Layer, Target, TRIM
from pathlib import Path
from datetime import datetime

from typing import Dict, List
from typing_extensions import Literal, Final

# Specify the directory of SRIM.exe
# For windows users the path will include C://...
SRIM_EXE_DIR: Final[Path] = Path(r"C:\users\nova\desktop\srim-2013")
MATERIAL: Final[Literal["ceria", "thoria", "urania"]] = "ceria"
ION_BEAM = "He"
energies: List[float] = [1.0e6, 2.0e6, 3.0e6]

ions: Dict[str, Ion] = {}
for energy in energies:
    key = f"{ION_BEAM}_{str(energy)[0]}_{str(energy)[1]}MeV"
    value = Ion(ION_BEAM, energy=energy)
    ions.update({key: value})

layer_ceria = Layer(
    {
        "Ce": {"stoich": 1.0, "E_d": 30.0, "lattice": 0.0, "surface": 3.0},
        "O": {"stoich": 2.0, "E_d": 30.0, "lattice": 0.0, "surface": 3.0},
    },
    density=8.9,
    width=20_000.0,  # depth? of layer in Angstrom 10_000 = 1 um
    name="ceria",
)

# Construct a target of a single layer of Nickel
if MATERIAL == "ceria":
    target = Target([layer_ceria])

save_time: Final[str] = datetime.now().strftime("%Y_%m_%d_%I%M%p")

# Initialize a TRIM calculation with given target and ion for 25 ions, quick calculation
for name, ion in ions.items():
    trim = TRIM(target, ion, number_ions=5, calculation=1)
    results = trim.run(SRIM_EXE_DIR)

    data_dir = Path(r".\data") / MATERIAL / save_time / name
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    TRIM.copy_output_files(SRIM_EXE_DIR, data_dir)
