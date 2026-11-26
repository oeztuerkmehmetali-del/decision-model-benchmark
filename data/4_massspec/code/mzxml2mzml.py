"""Minimal centroided mzXML -> MS2-only mzML (Sage reads mzML only). RT stays in minutes (pyteomics mzXML RT is minutes; psims scan_start_time is minutes).
python3 mzxml2mzml.py in.mzXML out.mzML
Keeps: scan number (as `scan=N` id), RT, precursor m/z + charge + intensity, HCD energy, centroid peaks. Drops MS1.
"""
import sys
from pyteomics import mzxml
from psims.mzml.writer import MzMLWriter

src, dst = sys.argv[1:3]
n_in = n_out = 0
with mzxml.read(src, use_index=False) as r, MzMLWriter(open(dst, "wb")) as w:
    w.controlled_vocabularies()
    w.file_description(["MSn spectrum"], [])
    w.software_list([{"id": "mzxml2mzml", "version": "1", "params": ["custom unreleased software tool"]}])
    w.instrument_configuration_list([w.InstrumentConfiguration(id="IC1", component_list=[], params=[])])
    w.data_processing_list([w.DataProcessing([w.ProcessingMethod(order=1, software_reference="mzxml2mzml", params=["Conversion to mzML"])], id="DP1")])
    with w.run(id="run1", instrument_configuration="IC1"):
        # spectrum count is unknown up front -> count MS2 in a cheap first pass is skipped; psims accepts any int, Sage ignores it
        with w.spectrum_list(count=55352):
            for s in r:
                n_in += 1
                if s["msLevel"] != 2 or "precursorMz" not in s:
                    continue
                p = s["precursorMz"][0]
                prec = {"mz": p["precursorMz"], "intensity": p.get("precursorIntensity", 0.0)}
                if "precursorCharge" in p:
                    prec["charge"] = int(p["precursorCharge"])
                w.write_spectrum(
                    s["m/z array"], s["intensity array"],
                    id=f"controllerType=0 controllerNumber=1 scan={s['num']}",
                    centroided=True, scan_start_time=float(s["retentionTime"]),
                    params=[{"ms level": 2}], precursor_information=prec,
                )
                n_out += 1
print(f"scans read {n_in}, MS2 written {n_out}", file=sys.stderr)
