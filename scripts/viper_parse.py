# Parse ViPER XML annotation files (AVSS2007 / i-LIDS format) into GT rows.
# Extracts, per clip: fps, resolution, the "put" frame, and the "abandoned" frame.
import xml.etree.ElementTree as ET
import glob, os, csv, re

NS = {"v": "http://lamp.cfar.umd.edu/viper#", "d": "http://lamp.cfar.umd.edu/viperdata#"}

def parse_one(path):
    """Return dict with clip, fps, w, h, put_frame, abandoned_frame, bag_box."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    # some ViPER files have a malformed tag (</config> instead of </descriptor>); repair before parse
    raw = raw.replace("</attribute>\n        </config>", "</attribute>\n        </descriptor>\n    </config>")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        # fallback: regex extraction if XML is too broken to parse
        return regex_fallback(raw, path)

    out = {"file": os.path.basename(path)}
    for sf in root.iter("{http://lamp.cfar.umd.edu/viper#}sourcefile"):
        out["clip"] = sf.get("filename")
    # Information attributes
    def find_val(name, kind):
        for a in root.iter("{http://lamp.cfar.umd.edu/viper#}attribute"):
            if a.get("name") == name:
                for child in a:
                    v = child.get("value")
                    if v is not None:
                        return v
        return None
    out["fps"] = float(find_val("FRAMERATE", "f") or 30.0)
    out["numframes"] = int(find_val("NUMFRAMES", "d") or 0)
    out["w"] = int(find_val("H-FRAME-SIZE", "d") or 0)
    out["h"] = int(find_val("V-FRAME-SIZE", "d") or 0)
    # objects: look for AbandonedObject and PutObject framespans
    put_f, aband_f, box = None, None, None
    for obj in root.iter("{http://lamp.cfar.umd.edu/viper#}object"):
        nm = obj.get("name"); span = obj.get("framespan")
        start = int(span.split(":")[0]) if span else None
        if nm == "PutObject": put_f = start
        if nm == "AbandonedObject":
            aband_f = start
            for bb in obj.iter("{http://lamp.cfar.umd.edu/viperdata#}bbox"):
                box = (int(bb.get("x")), int(bb.get("y")), int(bb.get("width")), int(bb.get("height")))
    out["put_frame"] = put_f
    out["abandoned_frame"] = aband_f
    out["bag_box"] = box
    return out

def regex_fallback(raw, path):
    out = {"file": os.path.basename(path)}
    m = re.search(r'filename="([^"]+)"', raw); out["clip"] = m.group(1) if m else None
    m = re.search(r'FRAMERATE.*?value="([\d.]+)"', raw, re.S); out["fps"] = float(m.group(1)) if m else 30.0
    m = re.search(r'AbandonedObject".*?framespan="(\d+):', raw, re.S)
    out["abandoned_frame"] = int(m.group(1)) if m else None
    m = re.search(r'PutObject".*?framespan="(\d+):', raw, re.S)
    out["put_frame"] = int(m.group(1)) if m else None
    out["bag_box"] = None; out["w"]=out["h"]=0; out["numframes"]=0
    return out

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(folder, "*.txt")) + glob.glob(os.path.join(folder, "*.xml")))
    print(f"found {len(files)} annotation files\n")
    rows = []
    for f in files:
        try:
            r = parse_one(f)
            bagsz = f"{r['bag_box'][2]}x{r['bag_box'][3]}px" if r.get("bag_box") else "?"
            print(f"{r.get('clip','?'):28s} fps={r['fps']} res={r['w']}x{r['h']} "
                  f"put={r['put_frame']} abandoned={r['abandoned_frame']} bag={bagsz}")
            rows.append(r)
        except Exception as e:
            print(f"ERROR on {f}: {e}")
    print(f"\nparsed {len(rows)} clips")
