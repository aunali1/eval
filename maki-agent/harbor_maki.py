"""Harbor installed-agent wrapper for maki (https://github.com/tontinton/maki).

Registered into the Harbor wheel by install-maki.sh as
``harbor.agents.installed.maki:Maki`` with ``AgentName.MAKI = "maki"`` so that
``harbor run -a maki`` parses and resolves.

Headless invocation inside the task container (working directory = task dir,
which is the environment exec default like every other installed agent):

    maki -p --yolo --trust --verbose --output-format stream-json \
        -m neuralwatt/kimi-k3 "<instruction>"

maki v0.5.5 emits Claude Code-compatible stream-json (system/init, assistant,
user, result envelopes), which this wrapper converts to an ATIF trajectory and
into AgentContext usage/cost fields. The eval pipeline reprices from
proxy-relay evidence, so cost population here is best-effort.

NEVER pass real API keys on the maki command line. NEURALWATT_API_KEY (a stub
placeholder; the egress proxy injects the real key for api.neuralwatt.com) is
forwarded from the runner process env / --agent-env into the container exec env.
"""

import json
import os
import re
import shlex
import uuid
from typing import Any, override

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


class Maki(BaseInstalledAgent):
    """maki coding agent (static musl binary, pinned at v0.5.5)."""

    SUPPORTS_ATIF: bool = True

    DEFAULT_MODEL = "neuralwatt/kimi-k3"
    DEFAULT_VERSION = "0.5.5"
    _OUTPUT_FILENAME = "maki.txt"
    _INSTALLER_URL = "https://maki.sh/install.sh"
    _INSTALL_DIR = "/usr/local/bin"

    # base64 of: maki.setup({ always_yolo = true, always_workflow = true,
    # always_thinking = "max" }). Evaluated harness defaults.
    _INIT_LUA_B64 = (
        "bWFraS5zZXR1cCh7CiAgICBhbHdheXNfeW9sbyA9IHRydWUsCiAgICBhbHdheXNfd29ya2Z"
        "sb3cgPSB0cnVlLAogICAgYWx3YXlzX3RoaW5raW5nID0gIm1heCIsCn0pCg=="
    )

    # base64 of providers.toml pinning the neuralwatt custom provider (OpenAI
    # protocol, api.neuralwatt.com, kimi-k3 1M ctx) so task containers never
    # need a models.dev catalog fetch.
    _PROVIDERS_TOML_B64 = (
        "W25ldXJhbHdhdHRdCmRpc3BsYXlfbmFtZSA9ICJOZXVyYWx3YXR0Igpwcm90b2NvbCA9ICJvcG"
        "VuYWkiCmJhc2VfdXJsID0gImh0dHBzOi8vYXBpLm5ldXJhbHdhdHQuY29tL3YxIgphcGlfa2V5X2"
        "VudiA9ICJORVVSQUxXQVRUX0FQSV9LRVkiCmRlZmF1bHRfbW9kZWwgPSAibmV1cmFsd2F0dC9raW1"
        "pLWszIgoKW1tuZXVyYWx3YXR0Lm1vZGVsc11dCmlkID0gImtpbWktazMiCnRpZXIgPSAic3Ryb25n"
        "Igpjb250ZXh0X3dpbmRvdyA9IDEwNDg1NjAKbWF4X291dHB1dF90b2tlbnMgPSAzMjc2OApzdXBw"
        "b3J0c190aGlua2luZyA9IHRydWUKc3VwcG9ydHNfdmlzaW9uID0gdHJ1ZQpwcmljaW5nX2lucHV0"
        "ID0gMy4wCnByaWNpbmdfb3V0cHV0ID0gMTUuMApwcmljaW5nX2NhY2hlX3JlYWQgPSAwLjMK"
    )

    def __init__(self, *args, version: str | None = None, **kwargs):
        super().__init__(*args, version=version or self.DEFAULT_VERSION, **kwargs)

    @staticmethod
    @override
    def name() -> str:
        # Matches AgentName.MAKI appended to harbor/models/agent/name.py by
        # install-maki.sh. Literal string keeps this module importable even
        # before the enum patch is applied.
        return "maki"

    @override
    def get_version_command(self) -> str | None:
        return "maki --version"

    @override
    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        match = re.search(r"(\d+\.\d+\.\d+)", text)
        if match:
            return match.group(1)
        return text

    def _release_tag(self) -> str:
        version = self._version or self.DEFAULT_VERSION
        return version if version.startswith("v") else f"v{version}"

    async def _installed_maki_satisfies_version(
        self, environment: BaseEnvironment
    ) -> bool:
        check = await environment.exec(command="command -v maki >/dev/null 2>&1")
        if check.return_code != 0:
            return False
        version_result = await environment.exec(command=self.get_version_command())
        if version_result.return_code != 0:
            return False
        installed = self.parse_version(version_result.stdout or "")
        return installed == (self._version or self.DEFAULT_VERSION)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if await self._installed_maki_satisfies_version(environment):
            self.logger.debug(
                "maki is already available at the requested version"
            )
            return

        await self.ensure_system_dependencies(environment, ("curl",))

        # Static musl binary; works in glibc containers. Download the pinned
        # release tarball from GitHub directly: maki.sh is not in the trial
        # egress allowlist, github.com is. Install to /usr/local/bin as root
        # so any agent user can run it.
        await self.exec_as_root(
            environment,
            command=(
                # POSIX sh only: task images vary (dash/alpine), no pipefail.
                "set -eu; "
                f"tag={self._release_tag()}; "
                'arch=$(uname -m); case "$arch" in '
                'x86_64|amd64) arch=x86_64;; aarch64|arm64) arch=aarch64;; '
                '*) echo "unsupported arch $arch" >&2; exit 1;; esac; '
                'url="https://github.com/tontinton/maki/releases/download/'
                '${tag}/maki-${tag}-${arch}-unknown-linux-musl.tar.gz"; '
                'tmp=$(mktemp -d); '
                'curl -fsSL "$url" -o "$tmp/maki.tar.gz" && '
                'tar xzf "$tmp/maki.tar.gz" -C "$tmp" && '
                f'install -m 0755 "$tmp/maki" {self._INSTALL_DIR}/maki && '
                'rm -rf "$tmp"; '
                "maki --version"
            ),
        )

        # Harness defaults for evaluation: workflow mode on (task callable
        # inside code_execution) and max thinking effort. Written for root
        # and every user with a home directory, since agent.user varies per
        # task image.
        await self.exec_as_root(
            environment,
            command=(
                "set -eu; "
                f"for home in /root /home/*; do "
                '  [ -d "$home" ] || continue; '
                '  mkdir -p "$home/.config/maki"; '
                f'  echo {self._INIT_LUA_B64!r} | base64 -d > "$home/.config/maki/init.lua"; '
                f'  echo {self._PROVIDERS_TOML_B64!r} | base64 -d > "$home/.config/maki/providers.toml"; '
                f"done"
            ),
        )

    def _runtime_env(self) -> dict[str, str]:
        env: dict[str, str] = {"NO_COLOR": "true"}
        # Forward every NEURALWATT_* var WITHOUT stripping the prefix
        # (BaseAgent._get_env_prefixed strips it; that is for prefixes like
        # OPENHANDS_ that map onto config keys, not provider auth env vars).
        # The API key value is a stub placeholder; the egress proxy injects
        # the real key for requests to api.neuralwatt.com.
        for source in (os.environ, self.extra_env):
            for key, value in source.items():
                if key.startswith("NEURALWATT_"):
                    env[key] = value
        return env

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or self.DEFAULT_MODEL
        env = self._runtime_env()

        # Pass the instruction via a uniquely-named env var to avoid any
        # shell-quoting issues in the command line. -p reads a piped stdin,
        # but the instruction here is a positional argument per the canonical
        # headless form, so stdin is /dev/null.
        instruction_shell_var = f"harbor_maki_instruction_{uuid.uuid4().hex}"
        instruction_env_var = instruction_shell_var.upper()
        run_env = {**env, instruction_env_var: instruction}
        output_path = (self.environment_logs_dir / self._OUTPUT_FILENAME).as_posix()

        await self.exec_as_agent(
            environment,
            command=(
                f'{instruction_shell_var}="${instruction_env_var}"; '
                f"unset {instruction_env_var}; "
                f"maki -p --yolo --trust --verbose "
                f"--output-format stream-json "
                f"-m {shlex.quote(model)} "
                f'"${instruction_shell_var}" '
                f"2>&1 </dev/null | tee {shlex.quote(output_path)}"
            ),
            env=run_env,
        )

        error = self._result_error()
        if error is not None:
            raise NonZeroAgentExitCodeError(f"maki reported an error: {error}")

    # -- stream-json parsing / context population ---------------------------

    def _parse_output_events(self) -> list[dict[str, Any]]:
        path = self.logs_dir / self._OUTPUT_FILENAME
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _result_error(self) -> str | None:
        for event in self._parse_output_events():
            if event.get("type") == "result" and event.get("is_error"):
                result_text = event.get("result")
                if isinstance(result_text, str) and result_text.strip():
                    return result_text.strip()
                return "maki result event reported is_error=true"
        return None

    @staticmethod
    def _build_metrics(usage: Any) -> Metrics | None:
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("input_tokens") or 0
        cached = usage.get("cache_read_input_tokens") or 0
        creation = usage.get("cache_creation_input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        if not (input_tokens or cached or creation or output_tokens):
            return None
        extra = {
            key: value
            for key, value in usage.items()
            if key
            not in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        }
        return Metrics(
            prompt_tokens=input_tokens + cached + creation,
            completion_tokens=output_tokens,
            cached_tokens=cached or None,
            cost_usd=usage.get("cost") or None,
            extra=extra or None,
        )

    @staticmethod
    def _extract_tool_result_content(content: Any) -> str | None:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if texts:
                return "\n".join(text for text in texts if text)
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    def _convert_events_to_trajectory(self, events: list[dict[str, Any]]) -> Trajectory | None:
        session_id: str | None = None
        model: str | None = self.model_name
        result_event: dict[str, Any] | None = None
        steps: list[Step] = []
        step_id = 1
        last_agent_step: Step | None = None
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        saw_usage = False

        for event in events:
            event_type = event.get("type")
            timestamp = event.get("timestamp")

            if event_type == "system":
                init = event.get("init") or event
                session_id = init.get("session_id") or session_id
                model = init.get("model") or model
                continue

            if event_type == "assistant":
                message = event.get("message") or {}
                msg_model = message.get("model") or model
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            text_parts.append(text)
                    elif block_type in ("thinking", "reasoning"):
                        thinking = block.get("thinking") or block.get("text")
                        if isinstance(thinking, str) and thinking:
                            reasoning_parts.append(thinking)
                    elif block_type == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                tool_call_id=str(block.get("id") or ""),
                                function_name=str(block.get("name") or ""),
                                arguments=block.get("input") or {},
                            )
                        )
                metrics = self._build_metrics(message.get("usage"))
                if metrics:
                    saw_usage = True
                    total_prompt += metrics.prompt_tokens or 0
                    total_completion += metrics.completion_tokens or 0
                    total_cached += metrics.cached_tokens or 0
                step = Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="agent",
                    model_name=msg_model,
                    message="\n".join(text_parts),
                    reasoning_content="\n\n".join(reasoning_parts) or None,
                    tool_calls=tool_calls or None,
                    metrics=metrics,
                    llm_call_count=1,
                )
                steps.append(step)
                last_agent_step = step
                step_id += 1
                continue

            if event_type == "user":
                message = event.get("message") or {}
                content = message.get("content")
                blocks = content if isinstance(content, list) else []
                tool_results = [
                    block
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                ]
                if tool_results:
                    results = [
                        ObservationResult(
                            source_call_id=str(block.get("tool_use_id") or "") or None,
                            content=self._extract_tool_result_content(
                                block.get("content")
                            ),
                        )
                        for block in tool_results
                    ]
                    if last_agent_step is not None and (
                        last_agent_step.observation is None
                        or not last_agent_step.observation.results
                    ):
                        last_agent_step.observation = Observation(results=results)
                    else:
                        steps.append(
                            Step(
                                step_id=step_id,
                                timestamp=timestamp,
                                source="user",
                                message="",
                                observation=Observation(results=results),
                            )
                        )
                        step_id += 1
                else:
                    if isinstance(content, str):
                        text = content
                    else:
                        text = "\n".join(
                            block.get("text", "")
                            for block in blocks
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    if text:
                        steps.append(
                            Step(
                                step_id=step_id,
                                timestamp=timestamp,
                                source="user",
                                message=text,
                            )
                        )
                        step_id += 1
                continue

            if event_type == "result":
                session_id = event.get("session_id") or session_id
                result_event = event
                continue

        if not steps:
            return None

        # Token totals: prefer the aggregate usage on the result event when
        # present; fall back to summing per-assistant usage.
        result_metrics = None
        if result_event:
            result_usage = result_event.get("usage") or {}
            if isinstance(result_usage, dict) and result_usage:
                input_tokens = result_usage.get("input_tokens") or 0
                cached = result_usage.get("cache_read_input_tokens") or 0
                creation = result_usage.get("cache_creation_input_tokens") or 0
                total_prompt = input_tokens + cached + creation
                total_cached = cached
                total_completion = result_usage.get("output_tokens") or 0
                saw_usage = True
            result_metrics = FinalMetrics(
                total_prompt_tokens=total_prompt if saw_usage else None,
                total_completion_tokens=total_completion if saw_usage else None,
                total_cached_tokens=total_cached or None,
                total_cost_usd=result_event.get("total_cost_usd"),
                total_steps=len(steps),
                extra={
                    key: value
                    for key, value in {
                        "duration_ms": result_event.get("duration_ms"),
                        "num_turns": result_event.get("num_turns"),
                        "is_error": result_event.get("is_error"),
                    }.items()
                    if value is not None
                }
                or None,
            )
        elif saw_usage:
            result_metrics = FinalMetrics(
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_cached_tokens=total_cached or None,
                total_steps=len(steps),
            )

        return Trajectory(
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=model,
            ),
            session_id=session_id or "unknown",
            steps=steps,
            final_metrics=result_metrics,
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._parse_output_events()
        if not events:
            self.logger.debug("No maki stream-json output found")
            return

        try:
            trajectory = self._convert_events_to_trajectory(events)
        except Exception as exc:
            self.logger.debug(f"Failed to convert maki events to trajectory: {exc}")
            return
        if trajectory is None:
            self.logger.debug("maki output contained no convertible steps")
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            with open(trajectory_path, "w", encoding="utf-8") as handle:
                json.dump(
                    trajectory.to_json_dict(), handle, indent=2, ensure_ascii=False
                )
            self.logger.debug(f"Wrote maki trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(f"Failed to write trajectory file {trajectory_path}: {exc}")

        metrics = trajectory.final_metrics
        if metrics:
            # Usage/cost are best-effort here; the eval pipeline reprices from
            # proxy-relay evidence.
            context.cost_usd = metrics.total_cost_usd
            context.n_input_tokens = metrics.total_prompt_tokens or 0
            context.n_cache_tokens = metrics.total_cached_tokens or 0
            context.n_output_tokens = metrics.total_completion_tokens or 0
            context.metadata = {
                **(context.metadata or {}),
                **{
                    "session_id": trajectory.session_id,
                    "model": trajectory.agent.model_name if trajectory.agent else None,
                    "num_steps": len(trajectory.steps),
                },
            }
