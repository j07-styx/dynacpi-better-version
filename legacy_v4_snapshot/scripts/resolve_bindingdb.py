import urllib.request,urllib.parse,re,html
from html.parser import HTMLParser
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):
  if tag=="a":
   d=dict(attrs)
   if "href" in d:self.urls.append(d["href"])
for u in ["https://www.bindingdb.org","https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"]:
 try:
  with urllib.request.urlopen(u,timeout=20) as r:t=r.read().decode(errors="replace");base=r.url
  l=Links();l.feed(t);print(base)
  print("\n".join(sorted(set(urllib.parse.urljoin(base,x) for x in l.urls if re.search(r"ownload|zip|PolicyNotice|zenodo",x,re.I)))[:60]))
 except Exception as e:print(repr(e))
