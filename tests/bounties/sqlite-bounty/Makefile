ARCH=$(shell uname -m)
VERSION=3.32.2
SOURCE_TAR=sqlite-version-$(VERSION).tar.gz
SOURCE_DIR=sqlite-version-$(VERSION)
BOUNTY_TAR=sqlite-$(VERSION)-bounty_$(ARCH).tar.xz
BOUNTY_RISCV64_TAR=sqlite-$(VERSION)-bounty_riscv64.tar.xz

CC=gcc
HARDEN_CFLAGS=-O1 -ftrivial-auto-var-init=pattern -D_FORTIFY_SOURCE=2 -fPIE -fstack-clash-protection -fstack-protector-strong -ffunction-sections -fdata-sections
HARDEN_LDFLAGS=-pie -Wl,-z,now,-z,relro,--gc-sections
SQLITE_CFLAGS=-DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION
SQLITE_LDFLAGS=-lm -s

TAR_FLAGS=--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime

# Use GitHub Actions cache when available
ifdef ACTIONS_RUNTIME_TOKEN
ifdef ACTIONS_RUNTIME_URL
ifdef BUILDX_BUILDER
SCOPE=sqlite-$(VERSION)-bounty
DOCKEROPTIONS=--cache-from type=gha,scope=$(SCOPE) --cache-to type=gha,mode=max,scope=$(SCOPE) --load
endif
endif
endif

all: $(BOUNTY_TAR)

riscv64: $(BOUNTY_RISCV64_TAR)

download: | $(SOURCE_DIR)

$(BOUNTY_TAR): sqlite3 start.sh
	tar $(TAR_FLAGS) -cJf $@ $^

sqlite3: $(SOURCE_DIR)/sqlite3.c
	$(CC) -o sqlite3 $(HARDEN_CFLAGS) $(SQLITE_CFLAGS) $(SOURCE_DIR)/shell.c $(SOURCE_DIR)/sqlite3.c $(HARDEN_LDFLAGS) $(SQLITE_LDFLAGS)

$(SOURCE_DIR)/sqlite3.c: $(SOURCE_DIR)/Makefile
	cd $(SOURCE_DIR) && $(MAKE) sqlite3.c

$(SOURCE_DIR)/Makefile: | $(SOURCE_DIR)
	cd $(SOURCE_DIR) && ./configure

$(SOURCE_DIR): $(SOURCE_TAR)
	tar xzf $(SOURCE_TAR)

$(SOURCE_TAR):
	wget -O $(SOURCE_TAR) https://github.com/sqlite/sqlite/archive/refs/tags/version-$(VERSION).tar.gz

ifneq ($(ARCH), riscv64)
$(BOUNTY_RISCV64_TAR): Dockerfile start.sh
	docker build --tag sqlite-bounty-cp --file Dockerfile --progress plain --build-arg VERSION=$(VERSION) $(DOCKEROPTIONS) .
	docker create --platform=linux/riscv64 --name sqlite-bounty-cp sqlite-bounty-cp
	docker cp sqlite-bounty-cp:/root/$@ $@
	docker rm sqlite-bounty-cp
	touch $@
endif

clean:
	rm -f sqlite3 $(SOURCE_DIR)-bounty_*.tar.xz
	if [ -f "$(SOURCE_DIR)/Makefile" ]; then $(MAKE) -C $(SOURCE_DIR) clean; fi

distclean: clean
	rm -rf sqlite-*

test-exploit:
	./start.sh exploit-sqlite-$(VERSION).sql
