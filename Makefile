# make install   -> $(BINDIR)/gpu_runner, a symlink to a copy of gpu_runner.py (+ gpu_runner_utils.py)
#                   in $(LIBDIR) that runs on its own uv venv in $(LIBDIR)/venv (independent of
#                   this checkout); re-run after changing either file; also installs $(LIBDIR)/gate, an authorized_keys forced command
#                   that only lets `gpu_runner submit|query|list|download|cancel` through
# make uninstall -> removes the command and its venv
# make install-mcp   -> gpu_runner_mcp.py in $(LIBDIR), run on the same venv, then asks which host
#                       runs the daemon (reached as `ssh HOST gpu_runner ...`), writes the [mcp]
#                       config and registers it with Claude Code
# make uninstall-mcp -> unregisters it and removes the script (the venv and config file are kept)
PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/share/gpu_runner
VENV = $(LIBDIR)/venv
PYTHON = $(VENV)/bin/python

.PHONY: install uninstall install-mcp uninstall-mcp

install: $(PYTHON)
	uv pip install --python $(PYTHON) 'rpyc>=6.0.2'
	install -d $(BINDIR)
	install -m 644 gpu_runner_utils.py $(LIBDIR)/gpu_runner_utils.py
	sed '1s|^#!.*|#!$(PYTHON)|' gpu_runner.py > $(LIBDIR)/gpu_runner.py
	chmod 755 $(LIBDIR)/gpu_runner.py
	# a symlink, not a copy: Python resolves it, so gpu_runner_utils is importable from $(LIBDIR)
	ln -sfn $(LIBDIR)/gpu_runner.py $(BINDIR)/gpu_runner
	sed -e '1s|^#!.*|#!$(PYTHON)|' -e 's|@RUNNER@|$(BINDIR)/gpu_runner|' gate.py > $(LIBDIR)/gate
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
	install -m 644 gpu_runner_utils.py $(LIBDIR)/gpu_runner_utils.py
	install -m 644 gpu_runner_mcp.py $(LIBDIR)/gpu_runner_mcp.py
	$(PYTHON) install_mcp.py --python $(PYTHON) --script $(LIBDIR)/gpu_runner_mcp.py

uninstall-mcp:
	-claude mcp remove -s user gpu_runner
	rm -f $(LIBDIR)/gpu_runner_mcp.py
