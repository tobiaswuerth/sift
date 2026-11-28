from collections.abc import Mapping
from types import MappingProxyType

from sift.interfaces.acts.jobs import JOBS
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.context import Command
from sift.interfaces.cli.shown import jobs as shown

COMMANDS: Mapping[str, Command] = MappingProxyType({"jobs": command(JOBS, shown.listed)})
