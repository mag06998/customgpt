
from pathlib import Path
import shutil

from_folder = Path(r"K:\Thermal\2 - Standards\ASHRAE\ASHRAE Fundamentals 2001")
to_folder = Path(__file__).parent.parent / "documents"

print(from_folder)
print(to_folder)

for file in from_folder.iterdir():
    if file.is_file():
        shutil.copy(file, to_folder / ("ashrae_fundamentals_"+str(file.name)))












