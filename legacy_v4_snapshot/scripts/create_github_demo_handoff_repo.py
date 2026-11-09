"""One-time authenticated GitHub repo creation for user-requested handoff.

Auth uses local Git Credential Manager. No tokens are logged, committed,
written to disk, or exposed in process arguments.
"""
import json
import os
import subprocess
import urllib.error
import urllib.request

expected_login="j07-styx"
repo_name="Demo-smth"
env=os.environ.copy()
env["GCM_INTERACTIVE"]="never"
env["GIT_TERMINAL_PROMPT"]="0"
proc=subprocess.run(["git","credential","fill"],
    input="protocol=https\nhost=github.com\n\n",
    text=True,capture_output=True,env=env,timeout=30)
if proc.returncode:
    raise RuntimeError("No noninteractive local Git credential was available")
cred=dict(line.split("=",1) for line in proc.stdout.splitlines() if "=" in line)
secret=cred.get("password")
if not secret:
    raise RuntimeError("Git Credential Manager returned no token")

def request(url,method="GET",data=None):
    raw=json.dumps(data).encode() if data is not None else None
    headers={"Authorization":"Bearer "+secret,
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
        "User-Agent":"DynaCPI-v4-Research-Handoff"}
    if raw is not None:headers["Content-Type"]="application/json"
    req=urllib.request.Request(url,method=method,data=raw,headers=headers)
    with urllib.request.urlopen(req,timeout=35) as r:
        return r.status,json.load(r)

status,user=request("https://api.github.com/user")
login=user.get("login","")
print(json.dumps({"credential_user":login,"matches_connected_account":login.casefold()==expected_login.casefold()}),flush=True)
if login.casefold()!=expected_login.casefold():
    raise RuntimeError("Local Git credentials belong to a different GitHub user; refusing to create repository")
url="https://api.github.com/repos/"+expected_login+"/"+repo_name
try:
    status,existing=request(url)
    if existing.get("full_name","").casefold()!=(expected_login+"/"+repo_name).casefold():
        raise RuntimeError("Unexpected repository response")
    print(json.dumps({"repository":existing["full_name"],
                      "created":False,"private":existing.get("private"),
                      "url":existing.get("html_url")}),flush=True)
except urllib.error.HTTPError as exc:
    if exc.code!=404:
        raise RuntimeError("Existing repository query failed: HTTP "+str(exc.code))
    status,new=request("https://api.github.com/user/repos",method="POST",
        data={"name":repo_name,"description":"DynaCPI v4 research implementation, experiment provenance, and AI-to-AI handoff (2026-11-05)",
              "private":True,"has_issues":True,"has_wiki":False,"auto_init":False})
    if status!=201:
        raise RuntimeError("Unexpected GitHub repository creation status")
    print(json.dumps({"repository":new.get("full_name"),
                      "created":True,"private":new.get("private"),
                      "url":new.get("html_url")}),flush=True)
