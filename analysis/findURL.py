import re    



def parse_api_url(url):
    match = re.search(r'https://api.github.com/repos/([^/]+)/([^/]+)/commits/([^/]+)', url)
    if match:
        name=match.group(1)
        repo=match.group(2)
        sha=match.group(3)
        return "git clone https://github.com/"+str(name)+"/"+str(repo)+".git   git checkout "+sha

print(parse_api_url(""))