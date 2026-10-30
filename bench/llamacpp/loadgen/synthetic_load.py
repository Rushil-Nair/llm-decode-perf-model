# Closed-loop load: U users, each sends its next request when the previous one finishes.
import json, random, sys, threading, time, urllib.request
url, users, total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
words = "the a of and to in is it you that he was for on are with as I his they be at one have this from".split()
lock = threading.Lock(); sent = [0]; toks = [0]
def worker(uid):
    r = random.Random(uid)
    while True:
        with lock:
            if sent[0] >= total: return
            sent[0] += 1
        body = {"prompt": " ".join(r.choice(words) for _ in range(r.randint(20, 60))),
                "n_predict": r.randint(64, 256), "ignore_eos": True, "temperature": 0, "cache_prompt": False}
        req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=600).read())
        with lock: toks[0] += out.get("tokens_predicted", 0)
t0 = time.time()
th = [threading.Thread(target=worker, args=(u,)) for u in range(users)]
[x.start() for x in th]; [x.join() for x in th]
dt = time.time() - t0
print(json.dumps({"users": users, "requests": total, "tokens": toks[0], "seconds": round(dt, 1), "tok_s": round(toks[0] / dt, 1)}))
