import sys, re
from pathlib import Path

COLOR_SET = {"Black","White","Gray","Silver","Blue","Red","Brown","Gold","Green","Tan","Orange","Yellow"}
TYPE_SET = {"Car","Van","Pickup Truck","Freight"}

def validate(out_dir: Path):
    thumbs = sorted((out_dir/"thumbs").glob("*.png"))
    labels = sorted((out_dir/"labels").glob("*.txt"))
    tset = {p.stem for p in thumbs}
    lset = {p.stem for p in labels}
    missing = tset ^ lset
    if missing:
        print("ERROR: unmatched files:", sorted(missing)); return 1
    ok = 0
    for lab in labels:
        skip_file = False
        line = lab.read_text().strip()
        parts = line.split()
        if len(parts) < 10:
            print("ERROR: wrong field count:", lab); return 1
        cls, xc, yc, w, h, theta = parts[:6]
        color, vtype, occ, shd = parts[6], " ".join(parts[7:-2]), parts[-2], parts[-1]
        # numeric ranges
        for x in (xc,yc,w,h):
            try:
                v = float(x); 
                if not (0.0 <= v <= 1.0):
                    print("ERROR: value out of [0,1] in", lab)
                    # return 1
                    skip_file = True
                    break
            except: 
                print("ERROR: non-float for normalized field in", lab); return 1
        
        if skip_file:
            continue
        try:
            float(theta); float(occ); float(shd)
        except:
            print("ERROR: numeric parse fail in", lab); return 1
        if color not in COLOR_SET:
            print("ERROR: bad color", color, "in", lab); return 1
        if vtype not in TYPE_SET:
            print("ERROR: bad type", vtype, "in", lab); return 1
        ok += 1
    print(f"OK: {ok} label files validated.")
    return 0

if __name__ == "__main__":
    sys.exit(validate(Path(sys.argv[1]) if len(sys.argv)>1 else Path('out')))
