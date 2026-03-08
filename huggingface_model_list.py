from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()

print(cache_info.cache_dir)

for repo in cache_info.repos:
    print(repo.repo_id)