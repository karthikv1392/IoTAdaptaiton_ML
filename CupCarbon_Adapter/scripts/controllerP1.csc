set p1Counter 0
loop
wait
read var
rdata $var sender val
print $sender
if($sender==s34)
	print $val
	plus p1Counter $p1Counter $val
	if($p1Counter>=200)
		send N 35
	else
		send A 35
	end
end
if($sender==s33)
	minus p1Counter $p1Counter $val
	if($p1Counter<200)
		send A 35
	else
		send N 35
	end
end
data p p1 $var
send $p 11