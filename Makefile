# make install   -> $(BINDIR)/gpu_runner, a symlink to a copy of src/main.py (+ the modules it imports)
#                   in $(LIBDIR) that runs on its own uv venv in $(LIBDIR)/venv (independent of
#                   this checkout); re-run after changing any of them; also installs $(LIBDIR)/gate, an authorized_keys forced command
#                   that only lets `gpu_runner submit|query|list|download|cancel` through
# make uninstall -> removes the command and its venv
# make install-mcp   -> mcp_server.py in $(LIBDIR), run on the same venv, then asks which host
#                       runs the daemon (reached as `ssh HOST gpu_runner ...`), writes the [mcp]
#                       config and registers it with Claude Code (MCP_FLAGS=--no-register to
#                       skip that, e.g. for jiuwenswarm only); prints the jiuwenswarm web UI JSON
# make uninstall-mcp -> unregisters it and removes the script (the venv and config file are kept)
PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/share/gpu_runner
VENV = $(LIBDIR)/venv
PYTHON = $(VENV)/bin/python
# imported by main.py
MODULES = $(addprefix src/,client.py config.py daemon.py job.py runner.py utils.py)

.PHONY: install uninstall install-mcp uninstall-mcp

install: $(PYTHON)
	uv pip install --python $(PYTHON) 'rpyc>=6.0.2'
	install -d $(BINDIR)
	install -m 644 $(MODULES) $(LIBDIR)/
	sed '1s|^#!.*|#!$(PYTHON)|' src/main.py > $(LIBDIR)/main.py
	chmod 755 $(LIBDIR)/main.py
	# a symlink, not a copy: Python resolves it, so the modules are importable from $(LIBDIR)
	ln -sfn $(LIBDIR)/main.py $(BINDIR)/gpu_runner
	sed -e '1s|^#!.*|#!$(PYTHON)|' -e 's|@RUNNER@|$(BINDIR)/gpu_runner|' src/gate.py > $(LIBDIR)/gate
	chmod 755 $(LIBDIR)/gate
	@echo "installed $(BINDIR)/gpu_runner (venv: $(VENV))"
	@echo "to restrict an ssh key to gpu_runner, prefix its line in ~/.ssh/authorized_keys with"
	@echo '  command="$(LIBDIR)/gate",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty'
	@case ":$$PATH:" in *":$(BINDIR):"*) ;; *) echo "note: $(BINDIR) is not on PATH";; esac

$(PYTHON):
	uv venv $(VENV)

uninstall:
	rm -f $(BINDIR)/gpu_runner
	rm -rf $(LIBDIR)

install-mcp: $(PYTHON)
	uv pip install --python $(PYTHON) 'mcp>=2.3,<3'
	install -m 644 src/utils.py $(LIBDIR)/utils.py
	install -m 644 src/mcp_server.py $(LIBDIR)/mcp_server.py
	$(PYTHON) src/install_mcp.py --python $(PYTHON) --script $(LIBDIR)/mcp_server.py $(MCP_FLAGS)

uninstall-mcp:
	-claude mcp remove -s user gpu_runner
	rm -f $(LIBDIR)/mcp_server.py
