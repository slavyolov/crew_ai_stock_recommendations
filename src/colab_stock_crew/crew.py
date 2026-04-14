from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from colab_stock_crew.tools.stock_tools import (
    StockSnapshotTool,
    SecFilingsTool,
    SecSectionTool,
    SerpApiSearchTool,
)


@CrewBase
class StockAnalysisCrew:
    agents: List[Agent]
    tasks: List[Task]

    def __init__(self):
        base = Path(__file__).resolve().parent
        with open(base / "config" / "agents.yaml", "r", encoding="utf-8") as f:
            self.agents_config = yaml.safe_load(f)
        with open(base / "config" / "tasks.yaml", "r", encoding="utf-8") as f:
            self.tasks_config = yaml.safe_load(f)
        self.local_llm = LLM(
            model="ollama/openhermes",
            base_url="http://localhost:11434"
        )

    @agent
    def market_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["market_researcher"],
            verbose=True,
            llm=self.local_llm,
            tools=[SerpApiSearchTool(), StockSnapshotTool(), SecFilingsTool()],
        )

    @agent
    def fundamentals_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamentals_analyst"],
            verbose=True,
            llm=self.local_llm,
            tools=[StockSnapshotTool(), SecFilingsTool(), SecSectionTool()],
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],
            verbose=True,
            llm=self.local_llm,
        )

    @task
    def market_research_task(self) -> Task:
        return Task(
            config=self.tasks_config["market_research_task"],
            agent=self.market_researcher(),
        )

    @task
    def fundamentals_task(self) -> Task:
        return Task(
            config=self.tasks_config["fundamentals_task"],
            agent=self.fundamentals_analyst(),
        )

    @task
    def investment_memo_task(self) -> Task:
        return Task(
            config=self.tasks_config["investment_memo_task"],
            agent=self.report_writer(),
            context=[self.market_research_task(), self.fundamentals_task()],
            output_file="stock_report.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            model="ollama/openhermes"
        )
