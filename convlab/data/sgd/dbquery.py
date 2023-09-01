import json
from typing import Dict, Union, Tuple


class Database:
    db = json.load(open("convlab/data/sgd/db/db2.json", "r"))
    db = json.loads(json.dumps(db).lower())
    values = {}
    for domain, entries in db.items():
        domain = domain.lower()
        if domain not in values:
            values[domain] = {}
        for entry in entries:
            for slot, value in entry.items():
                slot = slot.lower()
                value = value.lower()
                if slot not in values[domain]:
                    values[domain][slot] = set()
                values[domain][slot].add(value.lower())
        for slot in values[domain]:
            values[domain][slot] = sorted(values[domain][slot])

    def query(self, domain, constraints: Union[Dict[str, str], Tuple[str]]):
        domain = domain.lower()
        if isinstance(constraints, dict):
            constraints = constraints.items()
        found = []
        for record in self.db[domain]:
            record_keys = [k for k in record]
            for key, val in constraints:
                key, val = key.lower(), val.lower()
                if (
                    val == ""
                    or val == "dont care"
                    or val == "dontcare"
                    or val == "not mentioned"
                    or val == "don't care"
                    or val == "do n't care"
                    or key not in record_keys
                ):
                    continue
                else:
                    if key == "time" and domain != "ridesharing":
                        constraint_time = int(val.split(":")[0]) * 100 + int(
                            val.split(":")[1]
                        )
                        db_time = int(record[key].split(":")[0]) * 100 + int(
                            record[key].split(":")[1]
                        )
                        # assume db time <= constraint time
                        if constraint_time < db_time:
                            break
                    else:
                        if record[key].lower() != val.lower():
                            break
            else:
                found.append(record.copy())
        return found
