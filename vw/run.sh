set -e

clear

echo 'Format conversion...'
python vowpal-wabbit.py vw

echo ''
echo 'Training...'

if [ -f vw/train.vw.cache ]; then
    rm vw/train.vw.cache
fi

# vw vw/train.vw -c --passes 300 --ngram 7 -b 24 --ect 5 -f vw/train.model.vw > /dev/null
vw vw/train.vw -c -q nn --l1 0.000015 --passes 1000 --ngram 6 -b 24 --ect 5 --loss_function logistic -f vw/train.model.vw > /dev/null

#vw vw/train.vw -c --nn 200 -b 24 --passes 100 --ect 5 -f vw/train.model.vw

echo ''
echo 'Classifying C-V...'
vw vw/cv.vw -t -i vw/train.model.vw -p cv.preds.txt > /dev/null

echo ''
echo 'Checking C-V...'
Rscript check_preds.R vw/cv.vw.pred cv.preds.txt

echo ''
echo 'Classifying test...'
vw vw/test.vw -t -i vw/train.model.vw -p test.preds.txt > /dev/null

echo ''
echo 'Checking test...'
Rscript check_preds.R vw/test.vw.pred test.preds.txt

echo ''
echo 'Classifying pred...'
vw vw/pred.vw -t -i vw/train.model.vw -p pred.preds.txt > /dev/null

python vw_to_kaggle.py pred.preds.txt
