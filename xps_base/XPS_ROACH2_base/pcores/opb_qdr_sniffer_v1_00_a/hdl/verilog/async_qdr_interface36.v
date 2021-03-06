module async_qdr_interface36 #(
    parameter QDR_LATENCY = 10
  ) (
    input  host_clk,
    input  host_rst,
    input  host_en,
    input  host_rnw,
    input  [31:0] host_addr,
    input  [31:0] host_datai,
    input   [3:0] host_be,
    output [31:0] host_datao,
    output host_ack,
    input  qdr_clk,
    input  qdr_rst,
    output qdr_req,
    input  qdr_ack,
    output [31:0] qdr_addr,
    output qdr_r,
    output qdr_w,
    output [71:0] qdr_d,
    output  [7:0] qdr_be,
    input  [71:0] qdr_q,
    output sniffer_latch_out
  );


  /* TODO: A FIFO might be more appropriate but wastes a BlockRAM */

  /********** QDR assignments **********/

  /* On addressing:
     The LSB of the qdr address addresses
     A single burst ie 2 * 72 bits.
     The LSB of the OPB address addresses
     a single byte.

     An OPB word is 32 bits -> it takes
     4 cycles to write a complete 144 bit burst
  */

  reg [31:0] host_addr_reg;
  reg [31:0] host_datai_reg;
  reg  [3:0] host_be_reg;
  reg host_rnw_reg;

  reg [31:0] host_datao_pre;
  reg [31:0] host_datao_reg;
  
  reg second_cycle;  
  
  assign qdr_addr = host_addr_reg[31:4];   //burst address (16 byte words)
  wire [1:0] word_id = host_addr_reg[3:2]; //sub-address of the 32 bit word to re read/written

  assign qdr_r  = qdr_req & host_rnw_reg;
  assign qdr_w  = qdr_req & !host_rnw_reg;

  /* Write opb data to both upper and lower 36 bit halves of qdr data line.
   * Use byte enables to control which bits actually get written */
  //HACKHACKHACK assign qdr_d[71:0]  = {host_data_parity_ext, host_data_parity_ext};
  wire [7:0] extended_host_be_reg = word_id[0] ? {host_be_reg, 4'b0} : {4'b0, host_be_reg};
  assign qdr_be = (second_cycle && word_id[1]) || (!second_cycle && !word_id[1]) ?
                     extended_host_be_reg : 8'b0;

  /* Everything involving the write_buffer register is an almighty hack to make
   * the ROACH2 QDR CPU interface work without changing any katcp
   * code. The ROACH2 does not have access to the QDR byte-enable pins
   * so instead we keep track of 4x32 bit writes in a buffer. If this
   * buffer is written to the QDR on every write operation in a burst,
   * the final write in a group of 4 will leave the QDR in the state
   * required.
   * Sure, it would be neater to fill up a buffer over 4 OPB accesses 
   * and only then call a single write (like in the kat_svn code, but this requires no software
   * changes, and ultimately fewer OPB accesses. */

  reg [143:0] write_buffer;
  reg qdr_inputs_valid;
  /* Insert parity bits to pad to 36 bits */
  //wire [35:0] host_data_parity_ext = {1'b0, host_datai_reg[31:24], 1'b0, host_datai_reg[23:16], 1'b0, host_datai_reg[15:8], 1'b0, host_datai_reg[7:0]};
  wire [35:0] host_data_parity_ext = {4'b0000, host_datai_reg[31:0]};
  always @(posedge qdr_clk) begin
    //if (qdr_rst) begin
    //  write_buffer <= 144'b0;
    //end else if (qdr_inputs_valid) begin
      if (!host_rnw_reg) begin //only update the output registers on a write operation
        case (host_addr_reg[3:2])
          2'd0: begin
          write_buffer[ 35:0  ] <= host_data_parity_ext;
          end
          2'd1: begin
            write_buffer[ 71:36 ] <= host_data_parity_ext;
          end
          2'd2: begin
            write_buffer[107:72 ] <= host_data_parity_ext;
          end
          2'd3: begin
            write_buffer[143:108] <= host_data_parity_ext;
          end
        endcase
      end
    //end
  end

  assign qdr_d[71:0] = (second_cycle==1'b1) ? write_buffer[143:72] : write_buffer[71:0];

  reg [31:0] host_addr_reg_pre;
  reg [31:0] host_datai_reg_pre;
  reg [3:0] host_be_reg_pre;
  reg host_rnw_reg_pre;

  //register on inputs 
  always @(posedge host_clk) begin
      host_addr_reg_pre  <= host_addr;
      host_datai_reg_pre <= host_datai;
      host_be_reg_pre    <= host_be;
      host_rnw_reg_pre   <= host_rnw;
  end

  /* foo */

  reg trans_reg;
  reg trans_regR;
  reg trans_regRR;
  //synthesis attribute HU_SET of trans_regR  is SET1
  //synthesis attribute HU_SET of trans_regRR is SET1
  //synthesis attribute RLOC   of trans_regR  is X0Y0
  //synthesis attribute RLOC   of trans_regRR is X0Y1

  reg resp_reg;
  reg resp_regR;
  reg resp_regRR;
  //synthesis attribute HU_SET of resp_regR  is SET0
  //synthesis attribute HU_SET of resp_regRR is SET0
  //synthesis attribute RLOC   of resp_regR  is X0Y0
  //synthesis attribute RLOC   of resp_regRR is X0Y1

  reg wait_clear;

  always @(posedge host_clk) begin
    host_ack_reg <= 1'b0;

    resp_regR  <= resp_reg;
    resp_regRR <= resp_regR;

    if (host_rst) begin
      trans_reg  <= 1'b0;
      wait_clear <= 1'b0;
      host_ack_reg <= 1'b0;
    end else begin
      if (host_en) begin
        trans_reg  <= 1'b1;
        wait_clear <= 1'b0;
      end
      if (resp_regRR) begin
        trans_reg  <= 1'b0;
        wait_clear <= 1'b1;
      end
      if (wait_clear && !resp_regRR) begin
        wait_clear   <= 1'b0;
        host_ack_reg <= 1'b1;
      end
    end
  end

  //cross clock domain
  always @(posedge qdr_clk) begin
    if (trans_regRR) begin
      host_addr_reg  <= host_addr_reg_pre;
      host_datai_reg <= host_datai_reg_pre;
      host_be_reg    <= host_be_reg_pre;
      host_rnw_reg   <= host_rnw_reg_pre;
    end
  end

  reg qdr_trans_strb, qdr_resp_ready;
  reg [2:0] hshake_state;

  localparam RESP_IDLE       = 3'b001;
  localparam RESP_PREP_WRITE = 3'b010;
  localparam RESP_BUSY       = 3'b100;

  always @(posedge qdr_clk) begin
    qdr_trans_strb <= 1'b0;
    qdr_inputs_valid <= 1'b0; 

    trans_regR  <= trans_reg;
    trans_regRR <= trans_regR;
    if (qdr_rst) begin
      hshake_state <= RESP_IDLE;
      resp_reg   <= 1'b0;
    end else begin
      case (hshake_state)
        RESP_IDLE: begin
          if (trans_regRR) begin
            //qdr_trans_strb <= 1'b1;
            qdr_inputs_valid <= 1'b1;
            //host_addr_reg  <= host_addr;
            //host_datai_reg <= host_datai;
            //host_be_reg    <= host_be;
            //host_rnw_reg   <= host_rnw;
            hshake_state   <= RESP_PREP_WRITE;
            //hshake_state   <= RESP_BUSY;
          end
        end
        RESP_PREP_WRITE: begin
          qdr_trans_strb <= 1'b1;
          hshake_state <= RESP_BUSY;
        end
        RESP_BUSY: begin
          if (qdr_resp_ready)
            resp_reg  <= 1'b1;

          if (!trans_regRR) begin
            resp_reg  <= 1'b0;
            hshake_state   <= RESP_IDLE;
          end
        end
        default: begin
          hshake_state <= RESP_IDLE;
        end
      endcase
    end
  end
  
  /* Response Collection State Machine */

  reg [QDR_LATENCY - 1:0] qvld_shifter;

  reg [3:0] resp_state;
  localparam IDLE    = 4'b0001;
  localparam WAIT    = 4'b0010;
  localparam COLLECT = 4'b0100;
  localparam FINAL   = 4'b1000;
  reg sniffer_latch;

  always @(posedge qdr_clk) begin
    qvld_shifter   <= {qvld_shifter[QDR_LATENCY - 2:0], resp_state == WAIT && qdr_ack};
  end

  always @(posedge qdr_clk) begin
    qdr_resp_ready <= 1'b0;
    second_cycle   <= 1'b0;
    sniffer_latch  <= 1'b0;

    if (qdr_rst) begin
      resp_state <= IDLE;
    end else begin
      case (resp_state)
        IDLE: begin
          if (qdr_trans_strb) begin
            resp_state <= WAIT;
          end
        end
        WAIT: begin
          if (qdr_ack) begin
            second_cycle <= 1'b1;
            resp_state <= COLLECT;
          end
        end
        COLLECT: begin
          if (!host_rnw_reg) begin
            resp_state <= IDLE;
            qdr_resp_ready <= 1'b1;
          end else if (qvld_shifter[QDR_LATENCY-1]) begin
            if (!word_id[1]) begin
              resp_state <= IDLE;
              if(!word_id[0]) begin
                  //host_datao_reg <= {qdr_q[34:27], qdr_q[25:18], qdr_q[16:9], qdr_q[7:0]};
                  //host_datao_reg <= {qdr_q[31:0]};
                  sniffer_latch  <= 1'b1;
              end else begin
                  //host_datao_reg <= {qdr_q[70:63], qdr_q[61:54], qdr_q[52:45], qdr_q[43:36]};
                  //host_datao_reg <= {qdr_q[67:36]};
                  sniffer_latch  <= 1'b1;
              end
              qdr_resp_ready <= 1'b1;
            end else begin
              resp_state <= FINAL;
            end
          end
        end
        FINAL: begin
          qdr_resp_ready <= 1'b1;
          if(!word_id[0]) begin
              //host_datao_reg <= {qdr_q[34:27], qdr_q[25:18], qdr_q[16:9], qdr_q[7:0]};
              //host_datao_reg <= {qdr_q[31:0]};
              sniffer_latch  <= 1'b1;
          end else begin
              //host_datao_reg <= {qdr_q[70:63], qdr_q[61:54], qdr_q[52:45], qdr_q[43:36]};
              //host_datao_reg <= {qdr_q[67:36]};
              sniffer_latch  <= 1'b1;
          end
          resp_state <= IDLE;
        end
      endcase
    end
  end

//  assign host_datao = host_datao_reg;
  assign host_datao = host_datao_pre;
  reg host_ack_reg;
  assign host_ack = host_ack_reg;

  //cross clock domain
  always @(posedge host_clk) begin
    host_datao_pre <= host_datao_reg;
  end
 
  always @(posedge qdr_clk) begin
    case (resp_state)
      COLLECT: begin
        if(!word_id[0]) begin
          host_datao_reg <= {qdr_q[31:0]};
        end else begin
          host_datao_reg <= {qdr_q[67:36]};
        end
      end
      FINAL: begin
        if(!word_id[0]) begin
          host_datao_reg <= {qdr_q[31:0]};
        end else begin
          host_datao_reg <= {qdr_q[67:36]};
        end
      end
    endcase
  end
 
  assign qdr_req = qdr_trans_strb || resp_state == WAIT;
  assign sniffer_latch_out = sniffer_latch;

endmodule

