


# Main function
def main():
    # Setup logging
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    print(script_name)
    if (not setup_logging(console_log_output="stdout", console_log_level="debug", console_log_color=True,
                          logfile_file=script_name + ".log", logfile_log_level="debug", logfile_log_color=False,
                          log_line_template="%(color_on)s[%(created)d] [%(threadName)s] [%(levelname)-8s] %(message)s%(color_off)s")):
        print("Failed to setup logging, aborting.")
        return 1

    # Log some messages
    logging.debug("Debug message")
    logging.info("Info message")
    logging.warning("Warning message")
    logging.error("Error message")
    logging.critical("Critical message")


# Call main function
if __name__ == "__main__":
    sys.exit(main())
