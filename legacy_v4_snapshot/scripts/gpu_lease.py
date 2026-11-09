import msvcrt,time
from contextlib import contextmanager
@contextmanager
def gpu_lease(root):
 path=root/".gpu.lock"
 with path.open("a+b") as f:
  if not path.stat().st_size:f.write(b"1");f.flush()
  while True:
   f.seek(0)
   try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
   except OSError:time.sleep(2)
  try:yield
  finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
