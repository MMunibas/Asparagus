export my_orca=%orca%
rm -f run_orca_mp2.gbw run_orca_mp2.ges
$my_orca run_orca_mp2.inp > run_orca_mp2.out
python run_orca_mp2.py

