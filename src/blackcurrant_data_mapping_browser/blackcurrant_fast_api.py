from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List
import json
import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MapData:
    date: str
    filepath: str
    statistics: dict = None
    content: str = None

@dataclass
class TrendData:
    date: str
    condition: str
    percentage: float

class MapDataManager:
    def __init__(self, maps_directory: str, data_directory: str):
        self.maps_directory = Path(maps_directory)
        self.data_directory = Path(data_directory)
        self.maps: Dict[str, MapData] = {}
        self.trends: List[TrendData] = []
        self.last_scan_time = None
        self.scan_interval = timedelta(minutes=30)  # Rescan interval
        self._load_data()

    def _load_data(self):
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        for file_path in self.maps_directory.glob("*.html"):
            if match := re.search(date_pattern, file_path.name):
                date = match.group(0)
                if date in self.maps:
                    continue

                json_path = self.data_directory / f"{date}.json"
                statistics = None
                if json_path.exists():
                    with open(json_path) as f:
                        data = json.load(f)
                        total_counts = {"healthy": 0, "mineral": 0, "disease": 0}
                        for details in data.values():
                            for condition, count in details["counts"].items():
                                total_counts[condition] += count

                        total = sum(total_counts.values())
                        if total > 0:
                            statistics = {
                                condition: round(count / total * 100, 2)
                                for condition, count in total_counts.items()
                            }

                            # Add to trends
                            for condition, count in total_counts.items():
                                percentage = (count / total) * 100
                                self.trends.append(TrendData(
                                    date=date,
                                    condition=condition,
                                    percentage=round(percentage, 5)
                                ))

                self.maps[date] = MapData(
                    date=date,
                    filepath=str(file_path),
                    statistics=statistics
                )

    def refresh_data(self):
        """Refresh map data by rescanning directories if needed."""
        if not self.last_scan_time or datetime.now() - self.last_scan_time > self.scan_interval:
            self.last_scan_time = datetime.now()
            self._load_data()

    def get_map_data(self, date: str):
        self.refresh_data()
        if date not in self.maps:
            return None, None

        map_data = self.maps[date]
        if map_data.content is None:
            with open(map_data.filepath) as f:
                content = f.read()
            content = content.replace('<img ', '<img loading="lazy" ')
            content = content.replace('src="images/', 'src="/images/')
            content = content.replace('href="images/', 'href="/images/')
            map_data.content = content

        return map_data.content, map_data.statistics

    def get_available_dates(self):
        self.refresh_data()
        return sorted(self.maps.keys())

    def get_trends_data(self):
        self.refresh_data()
        return [
            {
                "Date": trend.date,
                "Condition": trend.condition,
                "Percentage": trend.percentage
            }
            for trend in sorted(self.trends, key=lambda x: x.date)
        ]


# FastAPI application setup
app = FastAPI()
data_manager = MapDataManager(
    maps_directory="heat_maps_for_server",
    data_directory="data_for_server"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/heat_maps_for_server", StaticFiles(directory="heat_maps_for_server"), name="heat_maps")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "map_files": {date: "" for date in data_manager.get_available_dates()}
        }
    )


@app.get("/get_map/{date}")
async def get_map(date: str):
    content, statistics = data_manager.get_map_data(date)
    if content is None:
        return HTMLResponse(content="Map not found", status_code=404)

    return JSONResponse({
        "html": content,
        "statistics": statistics
    })


@app.get("/get_trends")
async def get_trends():
    return {"data": data_manager.get_trends_data()}
