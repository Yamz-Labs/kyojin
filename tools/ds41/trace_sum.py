import csv, sys, collections
r = list(csv.DictReader(open(sys.argv[1])))
s = sorted(r, key=lambda x: int(x['Start_Timestamp']))
N = 8; W = float(sys.argv[2]) * N * 1e6
end = int(s[-1]['End_Timestamp']); w = [x for x in s if int(x['Start_Timestamp']) > end - W]
busy = sum(int(x['End_Timestamp']) - int(x['Start_Timestamp']) for x in w)
print(len(w) / N, "kernels/step; busy ms/step", busy / 1e6 / N, "wall", (int(w[-1]['End_Timestamp']) - int(w[0]['Start_Timestamp'])) / 1e6 / N)
c = collections.Counter(); n = collections.Counter()
for x in w: c[x['Kernel_Name'][:60]] += int(x['End_Timestamp']) - int(x['Start_Timestamp']); n[x['Kernel_Name'][:60]] += 1
for k, v in c.most_common(10): print(round(v / 1e6 / N, 3), n[k] / N, k)
