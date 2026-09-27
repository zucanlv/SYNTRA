import faiss
import pickle

index_dir = "/data/share/project/zucan/data/faiss/test_20260303_arguana"

index = faiss.read_index(f"{index_dir}/index.faiss")
print(f"[index.faiss] ntotal={index.ntotal}, d={index.d}, metric={index.metric_type}")

with open(f"{index_dir}/id_map.pkl", "rb") as f:
    id_map = pickle.load(f)
print(f"\n[id_map.pkl] type={type(id_map)}, len={len(id_map)}")
print("  first 5:", list(id_map.items())[:5] if isinstance(id_map, dict) else id_map[:5])

with open(f"{index_dir}/doc_dict.pkl", "rb") as f:
    doc_dict = pickle.load(f)
print(f"\n[doc_dict.pkl] type={type(doc_dict)}, len={len(doc_dict)}")
first_keys = list(doc_dict.keys())[:3]
for k in first_keys:
    print(f"  key={k!r}: {str(doc_dict[k])[:200]}")

with open(f"{index_dir}/lines_consumed.pkl", "rb") as f:
    lines_consumed = pickle.load(f)
print(f"\n[lines_consumed.pkl] type={type(lines_consumed)}, value={lines_consumed}")



'''

[index.faiss] ntotal=1280, d=1024, metric=0

[id_map.pkl] type=<class 'list'>, len=1280
  first 5: ['test-environment-aeghhgwpe-pro02b', 'test-environment-aeghhgwpe-pro02a', 'test-environment-aeghhgwpe-pro03b', 'test-environment-aeghhgwpe-pro01a', 'test-environment-aeghhgwpe-pro01b']

[doc_dict.pkl] type=<class 'dict'>, len=1280
  key='test-environment-aeghhgwpe-pro02b': {'title': 'animals environment general health health general weight philosophy ethics', 'text': "You don’t have to be vegetarian to be green. Many special environments have been created by livestock f
  key='test-environment-aeghhgwpe-pro02a': {'title': 'animals environment general health health general weight philosophy ethics', 'text': "Being vegetarian helps the environment  Becoming a vegetarian is an environmentally friendly thing to d
  key='test-environment-aeghhgwpe-pro03b': {'title': 'animals environment general health health general weight philosophy ethics', 'text': 'The key to good health is a balanced diet, not a meat- and fish-free diet. Meat and fish are good sourc

[lines_consumed.pkl] type=<class 'int'>, value=1280

'''