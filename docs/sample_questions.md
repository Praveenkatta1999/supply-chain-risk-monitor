# Sample questions for the root agent

Answers from `scripts/ask.py`. The agent chooses the sites and dates, runs the pipeline
once, and answers from the resulting brief; the brief itself is attached by Python.

## What are the risks to our chip supply this week?

Run `65381870a2f6`, asked on 2026-10-07.

**Answer.** For our chip supply this week, the highest identified risk is an unclear indirect threat to TSMC Hsinchu (S01) from a severe thunderstorm and flooding warning, reported with medium confidence. No risks were confirmed for our other chip-related sites, including the SK Hynix Icheon campus (S07), Intel Penang assembly and test (S09), and Infineon Dresden fab (S11).

**Sites and dates chosen.** S01, S07, S09, S11; 2026-10-01 to 2026-10-07. These four sites represent our primary chip fabrication, memory, power chip, and packaging suppliers, monitored over the last seven days to identify any active risks this week.

**Claims in the brief.**

- S01 (indirect, risk 1/25, unclear, medium confidence): A severe thunderstorm warning threatening heavy rain, lightning, and potential flooding was issued for six counties and cities. The warning advised of strong gusts, low visibility, and rising river levels in the affected areas. ([source 1](https://www.ettoday.net/news/20261002/3247800.htm))

**Model calls.** 30 in total: investigator 9, verifier 9, triage 4, root_agent 3, report_writer 2, reviewer 2, risk_scorer 1.
