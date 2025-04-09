export my_orca=$ORCA_COMMAND
rm -f run_orca.gbw run_orca.ges
$my_orca run_orca.inp > run_orca.out
python run_orca.py

