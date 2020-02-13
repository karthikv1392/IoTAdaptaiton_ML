set v1Counter 0
loop
wait
read var
rdata $var sender val
print $sender
if($sender==s1)
	plus v1Counter $v1Counter $val
	if($v1Counter>=500)
		send N 7
	else
		send A 7
	end
end
if($sender==s2)
	minus v1Counter $v1Counter $val
	if($v1Counter<500)
		send A 7
	else
		send N 7
	end
end
data p v1 $var
send $p 11

