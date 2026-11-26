"""GitHub pull request and CI operations for the optimizer supervisor."""
import json
import os
import re
import subprocess
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from auto_optimize_git import PublishBlocked, git


class LogRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        request = super().redirect_request(*args)
        if request:
            request.remove_header("Authorization")
        return request


class GitHub:
    def __init__(self, workspace):
        url = git(workspace, "remote", "get-url", "origin").decode().strip().removesuffix(".git")
        match = re.fullmatch(r"(?:git@github\.com:|https://github\.com/|ssh://git@github\.com/)([\w.-]+/[\w.-]+)", url)
        if not match:
            raise PublishBlocked("The optimizer needs a GitHub origin for pull requests.")
        self.repository = match[1]
        self.token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        if not self.token:
            credentials = subprocess.run(
                ["git", "credential", "fill"], cwd=workspace,
                input=b"protocol=https\nhost=github.com\n\n", capture_output=True, timeout=30,
                env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never"))
            fields = dict(line.split(b"=", 1) for line in credentials.stdout.splitlines() if b"=" in line)
            self.token = fields.get(b"password", b"").decode()
        if not self.token:
            raise PublishBlocked("Authenticate Git Credential Manager for github.com or set GH_TOKEN.")

    def api(self, path, method="GET", body=None):
        request = Request(f"https://api.github.com/repos/{self.repository}/{path}",
                          method=method, headers={"Authorization": "Bearer " + self.token,
                          "User-Agent": "vibrt-optimizer", "Content-Type": "application/json"},
                          data=json.dumps(body).encode() if body is not None else None)
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            raise PublishBlocked(f"GitHub {method} {path}: HTTP {exc.code}: "
                                 + exc.read(1500).decode(errors="replace")) from exc

    def pull(self, branch, result, message_path):
        owner = self.repository.split("/")[0]
        query = urlencode(dict(state="all", head=f"{owner}:{branch}", base="master"))
        existing = self.api("pulls?" + query)
        if existing:
            return self.api(f"pulls/{existing[0]['number']}")
        return self.api("pulls", "POST", dict(head=branch, base="master", title=result["commit_message"],
                        body=message_path.read_text(encoding="utf-8")))

    def pages(self, path, key=None):
        for page in range(1, 1001):
            data = self.api(f"{path}?per_page=100&page={page}")
            rows = data[key] if key else data
            yield from rows
            if len(rows) < 100:
                return
        raise PublishBlocked("Too many GitHub checks to verify.")

    def checks(self, head):
        runs = list(self.pages(f"commits/{head}/check-runs", "check_runs"))
        statuses = {}
        for row in self.pages(f"commits/{head}/statuses"):
            statuses.setdefault(row["context"], row)
        failed = []
        complete = any(r["name"] == "Renderer checks" and r["app"]["slug"] == "github-actions"
                       and r["conclusion"] == "success" for r in runs)
        for run in runs:
            if run["status"] != "completed":
                complete = False
            elif run["conclusion"] not in {"success", "neutral", "skipped"}:
                annotations = self.api(f"check-runs/{run['id']}/annotations?per_page=30")
                failure = dict(name=run["name"], conclusion=run["conclusion"], url=run["html_url"],
                               output=run["output"].get("summary"), annotations=annotations)
                if run["app"]["slug"] == "github-actions":
                    try:
                        failure["log_tail"] = self.job_log(run["id"])[-14000:]
                    except OSError as exc:
                        failure["log_error"] = type(exc).__name__
                failed.append(failure)
        for row in statuses.values():
            if row["state"] in {"error", "failure"}:
                failed.append(dict(name=row["context"], output=row["description"], url=row["target_url"]))
            elif row["state"] != "success":
                complete = False
        return complete and not failed, failed

    def job_log(self, job):
        request = Request(f"https://api.github.com/repos/{self.repository}/actions/jobs/{int(job)}/logs",
                          headers={"Authorization": "Bearer " + self.token, "User-Agent": "vibrt-optimizer"})
        with build_opener(LogRedirect()).open(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    def merge(self, number, head):
        result = self.api(f"pulls/{number}/merge", "PUT", dict(sha=head, merge_method="merge"))
        if not result.get("merged"):
            raise PublishBlocked("GitHub deferred the merge: " + str(result.get("message")))
