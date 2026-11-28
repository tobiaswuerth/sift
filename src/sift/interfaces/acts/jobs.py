from sift.api.jobs import JobsListed, ListJobs
from sift.interfaces.acts.acting import Act, Acting


def _jobs(acting: Acting, request: ListJobs) -> JobsListed:
    stopped = None if request.stop is None else acting.jobs.stopped(request.stop).id
    return JobsListed(acting.jobs.listed(), stopped)


JOBS = Act("jobs", ListJobs, JobsListed, _jobs)
