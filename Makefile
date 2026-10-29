# make install   -> $(BINDIR)/gpu_runner, a copy of gpu_runner.py that runs on its own uv venv
#                   in $(LIBDIR)/venv (independent of this checkout); re-run after changing
#                   gpu_runner.py
# make uninstall -> removes the command and its venv
# make install-mcp   -> gpu_runner_mcp.py in $(LIBDIR), run on the same venv, then asks which host
#                       runs the daemon, writes the [mcp] config and registers it with Claude Code
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
	sed '1s|^#!.*|#!$(PYTHON)|' gpu_runner.py > $(BINDIR)/gpu_runner
	chmod 755 $(BINDIR)/gpu_runner
	@echo "installed $(BINDIR)/gpu_runner (venv: $(VENV))"
	@case ":$$PATH:" in *":$(BINDIR):"*) ;; *) echo "note: $(BINDIR) is not on PATH";; esac

$(PYTHON):
	uv venv $(VENV)

uninstall:
	rm -f $(BINDIR)/gpu_runner
	rm -rf $(LIBDIR)

install-mcp: $(PYTHON)
	uv pip install --python $(PYTHON) 'mcp>=2.3,<3'
	install -m 644 gpu_runner_mcp.py $(LIBDIR)/gpu_runner_mcp.py
	$(PYTHON) install_mcp.py --python $(PYTHON) --script $(LIBDIR)/gpu_runner_mcp.py

uninstall-mcp:
	-claude mcp remove -s user gpu_runner
	rm -f $(LIBDIR)/gpu_runner_mcp.py
