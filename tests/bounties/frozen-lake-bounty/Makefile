TAR_FLAGS=--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime
BOUNTY_TAR=frozen-lake.tar.xz

$(BOUNTY_TAR): environment.py info.json test.py
	tar $(TAR_FLAGS) -cJf $@ $^
