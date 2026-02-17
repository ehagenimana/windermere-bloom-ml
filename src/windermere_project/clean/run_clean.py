{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d3e81f6-1d99-42ba-80f1-8a53b5dcfe74",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "from windermere_project.clean.builder import CleanDatasetBuilder, CleanConfig\n",
    "\n",
    "def main():\n",
    "    config = CleanConfig(\n",
    "        determinand_ids=(\"7887\",\"348\",\"9686\",\"111\",\"117\",\"9901\",\"76\",\"61\"),\n",
    "        datetime_col=\"phenomenonTime\",\n",
    "        site_col=\"samplingPoint.notation\",\n",
    "        determinand_col=\"determinand.notation\",\n",
    "        unit_col=\"unit\",\n",
    "        value_col=\"result\",\n",
    "        snapshot_id=\"raw_NW-88010013_ALLFULL_20260216T055951Z\",\n",
    "        coerce_numeric_errors=\"coerce\",\n",
    "        drop_non_numeric=True,\n",
    "    )\n",
    "\n",
    "    builder = CleanDatasetBuilder(config)\n",
    "\n",
    "    result = builder.build(\n",
    "        raw_snapshot_path=Path(\"data/raw/raw_NW-88010013_ALLFULL_20260216T055951Z.parquet\"),\n",
    "        output_dir=Path(\"data/clean\"),\n",
    "    )\n",
    "\n",
    "    print(result)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
