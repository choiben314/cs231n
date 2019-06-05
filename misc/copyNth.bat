n=0
for file in ./gl_train/rgb/*.png; do
   test $n -eq 0 && rm "$file"
   n=$((n+1))
   n=$((n%5))
done
