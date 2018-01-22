library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

library UNISIM;
use UNISIM.VComponents.all;

entity gpio_ext2simulink is
	 Port (
		gateway   : out std_logic_vector(0 downto 0);
		io_pad    : in  std_logic_vector(0 downto 0)
	 );
end gpio_ext2simulink;
architecture BEHAVIORAL of gpio_ext2simulink is
begin
    inportibuf_GEN: for i in 0 to (0) generate
        begin
            inportibuf: IBUF
            port map(
                I   => io_pad(i),
                O   => gateway(i)
            );
    end generate inportibuf_GEN;
end BEHAVIORAL;
