/**
 * Created by gabriel on 9/24/16.
 */

var exec = require('child_process').exec;

module.exports = function(command) {

  command = command.trim();
  var flag = null;

  if (command == "toggle-play") {
    flag = '-t';
  } else if (command == 'forward') {
    flag = '--fwd';
  } else if (command == 'backward') {
    flag = '--rew';
  } else if (command == 'play') {
    flag = '--play';
  } else if (command == 'stop') {
    flag = '--stop';
  }

  if (flag) {
    execute(flag);
  }

};

function execute(flag) {
  var cliCommand = "audacious " + flag;
  exec(cliCommand, function(err, stdout, stderr) {

    if (err) {
      console.error(err);
      return;
    }

    if (stdout && stdout.trim() != '') {
      console.log(stdout);
    }

    if (stderr && stderr.trim() != '') {
      console.log(stderr);
    }

  })
}