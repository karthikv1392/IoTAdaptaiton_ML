set p1Counter 0
set p2Counter 0
set v1Counter 0
set v2Counter 0
set v3Counter 0
loop
wait
read var
rdata $var sender val
function y adapter ada,100,10
if($y==10.0)
	data p s $var
	send $p 46
end
if($y==20.0)
	if($sender==s34)
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
	if($sender==s42)
		plus p2Counter $p2Counter $val
		if($p2Counter>=300)
			send N 43
		else
			send A 43
		end
	end
	if($sender==s41)
		minus p2Counter $p2Counter $val
		if($p2Counter<300)
			send A 43
		else
			send N 43
		end
	end
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
	if($sender==s24)
		plus v3Counter $v3Counter $val
		if($v3Counter>=300)
			send N 26
		else
			send A 26
		end
	end
	if($sender==s25)
		minus v3Counter $v3Counter $val
		if($v3Counter<300)
			send A 26
		else
			send N 26
		end
	end
	data p s $var
	send $p 46
end
if($y==30.0)
	data p s $var
	send $p 46
end

