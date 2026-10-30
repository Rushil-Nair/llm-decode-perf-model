"""Static SASS instructions per 32-weight group in the main loop of each gemv_sweep kernel.

The main loop is the smallest backward-branch loop that contains exactly ILP 128-bit global loads
(one per group). Its instruction count divided by ILP is what one lane executes per group.

    nvcc -O3 -std=c++17 -arch=sm_90 -cubin -o sweep.cubin kernels/gemv_sweep.cu
    cuobjdump -sass sweep.cubin > sweep.sass
    python microarch/sass_count.py sweep.sass results/sass_instr_sm90.json
"""
import re,collections,sys,json
s=open(sys.argv[1]).read()
out={}
for f in re.split(r"\n\s*Function : ",s)[1:]:
    name=f.split("\n")[0]
    m=re.search(r"gemv_(v\d)ILi(\d)E",name)
    if not m: continue
    k,ilp=m.group(1),int(m.group(2))
    ins=[(int(a,16),op,l) for l in f.split("\n") for a,op in re.findall(r"/\*([0-9a-f]{4})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)",l)]
    best=None
    for addr,op,l in ins:
        mm=re.search(r"BRA (0x[0-9a-f]+)",l)
        if mm and int(mm.group(1),16)<addr:
            lo=int(mm.group(1),16); body=[o for a,o,_ in ins if lo<=a<=addr]
            n128=sum(1 for o in body if o.startswith("LDG") and "128" in o)
            if n128==ilp and (best is None or len(body)<len(best)): best=body
    c=collections.Counter(o.split(".")[0] for o in best)
    out[f"{k}_ilp{ilp}"]={"instr_per_group":len(best)/ilp,"mix_per_group":{o:n/ilp for o,n in c.most_common(8)}}
    print(k,"ILP",ilp,"main loop",len(best),"-> per group %.1f"%(len(best)/ilp), {o:round(n/ilp,1) for o,n in c.most_common(7)})
if len(sys.argv)>2: json.dump(out,open(sys.argv[2],"w"),indent=1)
