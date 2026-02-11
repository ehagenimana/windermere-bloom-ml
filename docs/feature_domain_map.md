\# Feature Domain Map (Conceptual)



| Domain | Example variables | MVP priority | Notes |

|---|---|---|---|

| Lake chemistry | TP, SRP, nitrate, ammonium, DO, chlorophyll-a | Mandatory | Chemistry + oxygen dynamics linked to bloom risk |

| Physical lake conditions | water temperature, stratification proxies | Mandatory | If direct profiles unavailable, use proxies |

| Meteorology | air temperature, rainfall, wind speed/direction | Strongly recommended | Drives mixing + loading |

| Temporal structure | day-of-year, month, season | Mandatory | Captures seasonal ecology |

| Sampling structure | time since last obs, missingness indicators | Mandatory | Handles irregular monitoring |

| Site metadata | site ID, basin/arm (if exists) | Optional | Diagnostics; potential categorical feature |

| Remote sensing chlorophyll | satellite-derived chlorophyll (e.g., Sentinel) | Optional (fallback) | Used only if in-situ chlorophyll coverage insufficient |



