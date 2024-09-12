export my_orca=orca
rm -f run_orca.gbw run_orca.ges
$my_orca run_orca.inp > run_orca.out
python run_orca.py

