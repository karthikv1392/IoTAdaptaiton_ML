set mod 0
set counter 0
loop
if($mod==0)
	areadsensor var
	rdata $var t x sensorVal1
	data p s24 $sensorVal1
	plus counter $sensorVal1 $counter
	function y adapter ada,100,10
	if($y==10.0)
		send $p 11
	end
	if($y==20.0)
		send $p 11
	end
	if($y==30.0)
		send $p 51
	end
	while($sensorVal1<10.0)
		areadsensor var
		rdata $var t x sensorVal1
		data p s24 $sensorVal1
		plus counter $sensorVal1 $counter
		function y adapter ada,100,10
		if($y==10.0)
			if($counter>=300.0)
				send N 26
			else
				send A 26
			end
		send $p 11
		end
		if($y==20.0)
			send $p 11
		end
		if($y==30.0)
			send $p 51
		end
		delay 20000
	end
	if($sensorVal1>=10.0)
		set mod 1
	end
end
if($mod==1)
	while($sensorVal1>=10.0)
		areadsensor var
		rdata $var t x sensorVal1
		data p s24 $sensorVal1
		plus counter $sensorVal1 $counter
		function y adapter ada,100,10
		if($y==10.0)
			if($counter>=300.0)
				send N 26
			else
				send A 26
			end
		send $p 11
		end
		if($y==20.0)
			send $p 11
		end
		if($y==30.0)
			send $p 51
		end
		delay 10000
	end
	if($sensorVal1<10.0)
		set mod 0
	end
end
