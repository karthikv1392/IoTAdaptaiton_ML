set v2Counter 0
loop
wait
read var
rdata $var sender val
if($sender==s18)
	plus v2Counter $v2Counter $val
	if($v2Counter>=300)
		send N 17
	else
		send A 17
	end
end
if($sender==s20)
	minus v2Counter $v2Counter $val
	if($v2Counter<300)
		send A 17
	else
		send N 17
	end
end
data p v2 $var
send $p 11